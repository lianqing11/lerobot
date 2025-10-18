HF_USER=lianqing11
prefix=$1
rm -rf /Users/qinglian/.cache/huggingface/lerobot/lianqing11/record-train_${prefix}
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AB90657271 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AB90650101 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=false \
    --dataset.reset_time_s=10 \
    --dataset.repo_id=${HF_USER}/record-train_${prefix} \
    --dataset.num_episodes=10 \
    --dataset.single_task="Water the ${prefix} pot of flowers" \
    --dataset.push_to_hub=True \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
        wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, }" 