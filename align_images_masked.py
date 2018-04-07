import argparse
import cv2
import dlib
import json
import numpy
import skimage
from pathlib import Path
from tqdm import tqdm
from umeyama import umeyama

from face_alignment import FaceAlignment, LandmarksType

def faces_find(image, try_rotated):
    faces = FACE_ALIGNMENT.get_landmarks( image )
    if faces is None and try_rotated:
        angles = [90, 180, 270]
        for i,angle in enumerate(angles):
            image = image_rotate( image, -angle )
            faces = FACE_ALIGNMENT.get_landmarks( image )
            if faces and len( faces ) > 0:
                print( "Found face(s) with", str(angle), "degree rotation" )
                faces = faces_unrotate( faces, image, angle )
                break
    return faces

def faces_unrotate(faces, image, angle):
    for i,points in enumerate(faces):
        for ii,pointset in enumerate(points):
            if angle==90:
                faces[i][ii] = [pointset[1],image.shape[1]-pointset[0]]
            elif angle==180:
                faces[i][ii] = [image.shape[1]-pointset[0],image.shape[0]-pointset[1]]
            elif angle==270:
                faces[i][ii] = [image.shape[0]-pointset[1],pointset[0]]
    return faces

def image_resize(image, max_size):
    height, width = image.shape[:2]
    if max_size < height or max_size < width:
        scale = max_size / float(height)
        if max_size/float(width) < scale:
            scale = max_size / float(width)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image

def image_rotate(image, angle):
    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_matrix[0,0])
    abs_sin = abs(rotation_matrix[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_matrix[0, 2] += bound_w/2 - image_center[0]
    rotation_matrix[1, 2] += bound_h/2 - image_center[1]
    rotated_image = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))
    return rotated_image

def monkey_patch_face_detector(_):
    detector = dlib.get_frontal_face_detector()
    class Rect(object):
        def __init__(self,rect):
            self.rect=rect
    def detect( *args ):
        return [ Rect(x) for x in detector(*args) ]
    return detect

FACE_ALIGNMENT = FaceAlignment( LandmarksType._2D, enable_cuda=True, flip_input=False )

mean_face_x = numpy.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689 ])

mean_face_y = numpy.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182 ])

landmarks_2D = numpy.stack( [ mean_face_x, mean_face_y ], axis=1 )

def transform( image, mat, size, padding=0 ):
    mat = mat * size
    mat[:,2] += padding
    new_size = int( size + padding * 2 )
    return cv2.warpAffine( image, mat, ( new_size, new_size ) )

def main( args ):

    if args.use_gpu is False:
        dlib.cnn_face_detection_model_v1 = monkey_patch_face_detector

    input_dir = Path( args.input_dir )
    assert input_dir.is_dir()

    output_dir = input_dir / args.output_dir
    output_dir.mkdir( parents=True, exist_ok=True )

    output_file = input_dir / args.output_file

    input_files = list( input_dir.glob( "*." + args.file_type ) )

    if args.maxFrames > 0:
      input_files=input_files[args.startFrame:args.startFrame+args.maxFrames]
    elif args.startFrame>0:
      input_files=input_files[args.startFrame:]

    assert len( input_files ) > 0, "Can't find input files"

    def iter_face_alignments():
        for fn in tqdm( input_files ):
            
            image = cv2.imread( str(fn) )
            scale = 1

            if image is None:
              tqdm.write( "Can't read image file: ", fn )
              continue

            if args.fast_mode:
                image_sm = image_resize(image, args.fast_mode_size)
                scale = image_sm.shape[1]/image.shape[1]
                faces = faces_find( image_sm.copy(), args.try_rotated )
            else:
                faces = faces_find( image.copy(), args.try_rotated )
						
            if faces is None: continue
            if len(faces) == 0: continue
            if args.only_one_face and len(faces) != 1: continue

            for i,points in enumerate(faces):
                points = points/scale
                alignment = umeyama( points[17:], landmarks_2D, True )[0:2]
                aligned_image = transform( image, alignment, 160, 48 )

                if len(faces) == 1:
                    out_fn = "{}.jpg".format( Path(fn).stem )
                else:
                    out_fn = "{}_{}.jpg".format( Path(fn).stem, i )

                out_fn = output_dir / out_fn
                cv2.imwrite( str(out_fn), aligned_image )

                yield str(fn.relative_to(input_dir)), str(out_fn.relative_to(input_dir)), list( alignment.ravel() ), list(points.flatten().astype(float))

    face_alignments = list( iter_face_alignments() )

    with output_file.open('w') as f:
        results = json.dumps( face_alignments, ensure_ascii=False )
        f.write( results )

    print( "Save face alignments to output file:", output_file )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "input_dir"  , type=str )
    parser.add_argument( "output_dir" , type=str, nargs='?', default='aligned' )
    parser.add_argument( "output_file", type=str, nargs='?', default='alignments.json' )

    parser.set_defaults( only_one_face=False )
    parser.add_argument('--one-face' , dest='only_one_face', action='store_true'  )
    parser.add_argument('--all-faces', dest='only_one_face', action='store_false' )
    
    parser.add_argument( "--startFrame", type=int, default='0' )
    parser.add_argument( "--maxFrames", type=int, default='0' )
    
    parser.add_argument( "--file-type", type=str, default='jpg' )

    parser.set_defaults( fast_mode=False )
    parser.add_argument( '--fast-mode' , dest='fast_mode', action='store_true' )

    parser.add_argument( "--fast-mode-size", type=int, default='800' )

    parser.set_defaults( try_rotated=False )
    parser.add_argument( '--try-rotated' , dest='try_rotated', action='store_true' )

    parser.set_defaults( use_gpu=False )
    parser.add_argument( '--use-gpu' , dest='use_gpu', action='store_true' )

    main( parser.parse_args() )
