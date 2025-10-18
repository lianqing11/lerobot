# lerobot-calibrate \
#     --robot.type=so101_follower \
#     --robot.port=/dev/tty.usbmodem5AB90657271 \
#     --robot.id=my_awesome_follower_arm

# lerobot-calibrate \
#     --robot.type=so101_leader \
#     --robot.port=/dev/tty.usbmodem5AB90650101 \
#     --robot.id=my_awesome_leader_arm

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AB90650101 \
    --teleop.id=my_awesome_leader_arm # <- Give the robot a unique name