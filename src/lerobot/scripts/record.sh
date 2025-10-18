HF_USER=reedee123
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AB90657271 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AB90650101 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=false \
    --dataset.repo_id=${HF_USER}/record-train4 \
    --dataset.num_episodes=10 \
    --dataset.single_task="Water the first pot of flowers" \
    --dataset.push_to_hub=False \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \