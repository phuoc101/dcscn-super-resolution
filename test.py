import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import cv2
import time
import numpy as np
import DCSCN
from helper import loader, args, utilty as util

args.flags.DEFINE_boolean("save_results", True, "Save result, bicubic and loss images.")
args.flags.DEFINE_boolean("compute_bicubic", False, "Compute bicubic performance.")

FLAGS = args.get()

def main():
    # if len(not_parsed_args) > 1:
    #     print("Unknown args:%s" % not_parsed_args)
    #     exit()
    #
    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    if (FLAGS.frozenInference):
        model.load_graph(FLAGS.frozen_graph_path)
        model.build_summary_saver(with_saver=False) # no need because we are not saving any variables
    else:
        model.build_graph()
        model.build_summary_saver()
    model.init_all_variables()
    # if FLAGS.test_dataset == "all":
    #     test_list = ['set5', 'set14', 'bsd100']
    # else:
    #     test_list = [FLAGS.test_dataset]

    for i in range(FLAGS.tests):
        if (not FLAGS.frozenInference):
            model.load_model(FLAGS.load_model_name, trial=i, output_log=True if FLAGS.tests > 1 else False)

        if FLAGS.compute_bicubic:
            for test_data in test_list:
                print(test_data)
                evaluate_bicubic(model, test_data)

        # for test_data in test_list:
        #     evaluate_model(model, test_data)
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if ret:
            # true_image = loader.build_input_image(frame, channels=FLAGS.channels, scale=FLAGS.scale,
            #                                        alignment=FLAGS.scale)
            true_image = util.set_image_alignment(frame, FLAGS.scale)
            input_y_image = util.convert_rgb_to_y(true_image)
            output_y_image = model.do(input_y_image).astype(np.uint8)
            cv2.imshow("orig", true_image.astype(np.uint8))
            cv2.imshow("output y", output_y_image)
            scaled_ycbcr_image = util.convert_rgb_to_ycbcr(
                util.resize_image_by_pil(true_image, FLAGS.scale)).astype(np.uint8)
            hr_image = util.convert_y_and_cbcr_to_rgb(output_y_image, scaled_ycbcr_image[:, :, 1:3]).astype(np.uint8)
            cv2.imshow("sisr", hr_image)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            pass

if __name__ == "__main__":
    main()
    
