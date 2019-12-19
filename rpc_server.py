import base64
import re

import cv2
import pika
import json

from app2 import handle_colorization, from_png_to_jpg, np

server_ip = "192.168.88.234"
queue_name = "iAnimeColorizationQueue"

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=server_ip))

channel = connection.channel()

channel.queue_declare(queue=queue_name)


def get_request_image(image):
    image = re.sub('^data:image/.+;base64,', '', image)
    image = base64.urlsafe_b64decode(image)
    image = np.fromstring(image, dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image


def on_request(ch, method, props, body):
    try:
        task = json.loads(body, encoding='utf-8')
        image_req = get_request_image(task['image'])
        jpg_image = from_png_to_jpg(image_req)
        path = './colorize/' + task['receipt'] + '.png'

        pool = [jpg_image, task['points'], path]

        print("开始处理...")
        handle_colorization(pool)
        with open(path, 'rb') as f:
            res = base64.b64encode(f.read())
            res = str(res)[2:-1]
            response = json.dumps({'StatusCode': 0,
                                   'image': res,
                                   'receipt': task['receipt']})
    except object as e:
        print("导出失败")
        print(e)
        response = json.dumps({'StatusCode': -1})

    print("处理完成!")

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queue_name, on_message_callback=on_request)

print(" [x] Awaiting Colorize requests")
channel.start_consuming()
