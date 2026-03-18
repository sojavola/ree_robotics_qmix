#!/bin/bash
# ============================================================================
# REE Exploration Visualization Runner
# ============================================================================
# Script pour lancer la visualisation RViz2 du systeme REE
# Usage: ./run_visualization.sh [--server] [--viz-only]
# ============================================================================

set -e

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   REE Exploration RViz2 Visualization     ${NC}"
echo -e "${BLUE}============================================${NC}"

# Aller au workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE_DIR/src"

# Source ROS2
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}[OK] ROS2 Humble sourced${NC}"
elif [ -f "/opt/ros/iron/setup.bash" ]; then
    source /opt/ros/iron/setup.bash
    echo -e "${GREEN}[OK] ROS2 Iron sourced${NC}"
else
    echo -e "${RED}[ERROR] ROS2 not found!${NC}"
    exit 1
fi

# Source workspace local
if [ -f "$WORKSPACE_DIR/src/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/src/install/setup.bash"
    echo -e "${GREEN}[OK] Workspace sourced${NC}"
else
    echo -e "${YELLOW}[WARN] Workspace not built. Building now...${NC}"
    cd "$WORKSPACE_DIR/src"
    colcon build --symlink-install
    source install/setup.bash
fi

# Parser les arguments
RUN_SERVER=false
VIZ_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            RUN_SERVER=true
            shift
            ;;
        --viz-only)
            VIZ_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--server] [--viz-only]"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${YELLOW}Options:${NC}"
echo "  Run Server: $RUN_SERVER"
echo "  Viz Only:   $VIZ_ONLY"
echo ""

# Lancer les nodes
if [ "$VIZ_ONLY" = true ]; then
    echo -e "${GREEN}[INFO] Starting visualization only...${NC}"
    ros2 launch ree_exploration_viz visualization.launch.py
elif [ "$RUN_SERVER" = true ]; then
    echo -e "${GREEN}[INFO] Starting full system (server + visualization)...${NC}"
    ros2 launch ree_exploration_viz full_system.launch.py
else
    echo -e "${GREEN}[INFO] Starting visualization nodes...${NC}"
    echo -e "${YELLOW}[TIP] Run the server separately with: ros2 run ree_exploration_server server_node${NC}"
    echo ""

    # Lancer les nodes de visualisation en parallele
    ros2 run ree_exploration_viz visualization_node &
    VIZ_PID=$!

    ros2 run ree_exploration_viz robot_marker_publisher &
    ROBOT_PID=$!

    ros2 run ree_exploration_viz mineral_heatmap_publisher &
    HEATMAP_PID=$!

    # Lancer RViz2 avec config
    sleep 2
    ros2 run rviz2 rviz2 -d "$WORKSPACE_DIR/src/ree_exploration_viz/config/ree_exploration.rviz" &
    RVIZ_PID=$!

    echo ""
    echo -e "${GREEN}[OK] All nodes started${NC}"
    echo "  Visualization Node PID: $VIZ_PID"
    echo "  Robot Marker PID:       $ROBOT_PID"
    echo "  Heatmap PID:            $HEATMAP_PID"
    echo "  RViz2 PID:              $RVIZ_PID"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all nodes${NC}"

    # Attendre
    trap "kill $VIZ_PID $ROBOT_PID $HEATMAP_PID $RVIZ_PID 2>/dev/null" EXIT
    wait
fi
