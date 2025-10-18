# lerobot-teleoperate \
# --robot.type=so101_follower \
# --robot.port=/dev/tty.usbmodem5AB90650101 \
# --robot.id=my_awesome_follower_arm \
# --teleop.type=so101_leader \
# --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
# --teleop.port=/dev/tty.usbmodem5AB90657271 \
# --teleop.id=my_awesome_leader_arm \
# --display_data=true


lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AB90650101 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AB90657271 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=seeedstudio123/test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the black cube" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30 i