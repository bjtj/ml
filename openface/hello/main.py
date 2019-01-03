import cv2
import dlib
import openface
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dlib-align-weight', type=str,
                        default='../../../openface/models/dlib/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--openface-weight', type=str,
                        default='../../../openface/models/openface/nn4.small2.v1.t7')

    args = parser.parse_args()

    align_weight_path = args.dlib_align_weight
    openface_weight_path = args.openface_weight
    use_cuda = True

    dlib_detector = dlib.get_frontal_face_detector()
    alignnet = openface.AlignDlib(align_weight_path)
    facenet = openface.TorchNeuralNet(openface_weight_path, cuda = use_cuda)

    cam = cv2.VideoCapture(0)


    while True:
        ret, frame = cam.read()
        bbs = dlib_detector(frame, 1)
        for bb in bbs:
            face = alignnet.align(96, frame, bb,
                                  landmarkIndices = openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
            cv2.imshow('face', face)
            rep = facenet.forward(face)
            cv2.rectangle(frame,
                          (bb.left(), bb.top()),
                          (bb.right(), bb.bottom()),
                          (255, 255, 255), 2)

        cv2.imshow('preview', frame)


        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    print('Done')

if __name__ == '__main__':
    main()
