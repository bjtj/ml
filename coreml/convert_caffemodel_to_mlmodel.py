import coremltools
import sys


def convert(caffemodel_path, prototxt_path, mlmodel_path):
    coreml_model = coremltools.converters.caffe.convert((caffemodel_path, prototxt_path),
                                                        image_input_names="data", is_bgr=True)
    coreml_model.save(mlmodel_path)


def main():
    caffemodel_path = sys.argv[1]
    prototxt_path = sys.argv[2]
    mlmodel_path = sys.argv[3]
    convert(caffemodel_path, prototxt_path, mlmodel_path)

if __name__ == '__main__':
    main()
