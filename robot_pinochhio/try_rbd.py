import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from xml.etree.ElementTree import XML, fromstring

class PinocchioTest(Node):

    def __init__(self):
        super().__init__('pinocchio_test_node')
        self.subscription = self.create_subscription(String,'/robot_description',self.listener_callback,QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,depth=1))
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        #self.get_logger().info('I heard: "%s"' % msg.data)
        print(fromstring(msg.data))


def main(args=None):
    rclpy.init(args=args)

    pin_test = PinocchioTest()

    rclpy.spin(pin_test)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pin_test.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
