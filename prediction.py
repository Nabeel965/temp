from faceparsing.logger import setup_logger
from faceparsing.model import BiSeNet

import torch
from pathlib import Path

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from PIL import Image
from psgan import Inference

import faceutils as futils
from psgan import PostProcess
from setup import setup_config, setup_argparser
import warnings
warnings.filterwarnings("ignore")

def vis_parsing_maps(im, parsing_anno, stride):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) #+ 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, 16):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = [255,255,255]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    return vis_im

def evaluate(args, mode=0, cp='model_final_diss.pth'):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    if args.device=='cpu':
      net=net.cpu()
      net.load_state_dict(torch.load(cp,map_location=torch.device('cpu')))
    else:
      net=net.cuda()
      net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if mode==1:
      cap = cv2.VideoCapture(args.inputpath)

      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      fps = cap.get(cv2.CAP_PROP_FPS) 
      outwriter =cv2.VideoWriter('../results/out'+args.inputpath.split('/')[-1],
                              cv2.VideoWriter_fourcc(*"MJPG"),  fps, (width,height))
      #outwriter = cv2.VideoWriter('out'+args.inputpath.split('/')[-1], -1, fps, (width,height))
    #framecount=0
    frame=0
    if mode ==2:
      cap = cv2.VideoCapture(0)
    if mode>0:
      notimg=cap.isOpened()
    if mode==0:
      frame=cv2.imread(args.inputpath)
      notimg=False
    
    while(  notimg or frame is not None):
      if mode>0:
        ret, frame = cap.read() 
      #framecount+=1
      if frame is None:# or framecount>10:
        break
      frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      if args.useseg:
        with torch.no_grad():
              #original_source = Image.open(imgpath)
              img = to_tensor(frame)
              img = torch.unsqueeze(img, 0)
              if args.device=='cuda':
                img = img.cuda()
              out = net(img)[0]
              parsing = out.squeeze(0).cpu().numpy().argmax(0)

              vis_img=vis_parsing_maps(frame, parsing, stride=1)
              vis_img=cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
              ret,vis_img = cv2.threshold(vis_img,170,255,cv2.THRESH_BINARY)
              num_labels ,labels,stats ,centroids  = cv2.connectedComponentsWithStats(vis_img)
              indices=np.where(stats[:,4]>100*100)
              stats=stats[indices]
              objects=stats.shape[0]-1
      else:
              objects=1

      parser = setup_argparser()
      config = setup_config(args)

      inference = Inference(
          config, args.device, args.model_path)
      postprocess = PostProcess(config)
      temp_img=np.array(frame)
      temp=np.array(frame).copy()
      new_face_swapped=np.array(frame).copy()
      for i in range(1,objects+1):

        if args.useseg:
          maxwh=50+max(stats[i,2],stats[i,3])
          source=temp_img[ -50+stats[i,1]:stats[i,1]+maxwh , -50+stats[i,0]:stats[i,0]+maxwh ]
        else:
          source=frame
        source_shape=source.shape
        source=Image.fromarray(source).convert('RGB')
        if min(source_shape[1],source_shape[0])<300:
          Upsampled=1
          source=source.resize((source_shape[1]*2,source_shape[0]*2), Image.BILINEAR)
        else:
          Upsampled=0
        reference_paths = list(Path(args.reference_dir).glob("*"))
        np.random.shuffle(reference_paths)
        for reference_path in reference_paths:
            if not reference_path.is_file():
                print(reference_path, "is not a valid file.")
                continue

            reference = Image.open(reference_path).convert("RGB")

            image, face = inference.transfer(source, reference, with_face=True)
            if face is None:
              break

            source_crop = source.crop(
                (face.left(), face.top(), face.right(), face.bottom()))
            if Upsampled==1:
              source=source.resize((source_shape[1],source_shape[0]), Image.BILINEAR)

            face_swapped=np.array(source).copy()
            image = postprocess(source_crop, image)
            if Upsampled==1:
              image=image.resize((int(image.size[1]/2),int(image.size[0]/2)), Image.BILINEAR)
              top=int(face.top()/2)
              left=int(face.left()/2)
            else:
              top=face.top()
              left=face.left()         
            bottom=image.size[1]+top
            right=image.size[0]+left
            face_swapped[top:bottom,left:right]=np.array(image)#*1.1
          
            kernel = np.ones((5,5),np.float32)/25
            face_swapped[-10+bottom:50+bottom,left:right]= cv2.filter2D(face_swapped[-10+bottom:50+bottom,left:right],-1,kernel)
            if args.useseg:
              temp[ -50+stats[i,1]:stats[i,1]+maxwh ,
                  -50+stats[i,0]:stats[i,0]+maxwh ]=face_swapped
              new_face_swapped[labels==indices[0][i]]=temp[labels==indices[0][i]]
            else:

              new_face_swapped=face_swapped.copy()
      if mode==0:
        new_face_swapped=Image.fromarray(new_face_swapped)
        new_face_swapped.save('../results/out'+args.inputpath.split('/')[-1])   
        frame=None     
      if mode>0:
        new_face_swapped=cv2.cvtColor(new_face_swapped,cv2.COLOR_RGB2BGR)
        notimg=cap.isOpened()
      if mode==1:
        outwriter.write(new_face_swapped)
      if mode==2:
        cv2.imshow('Prediction',new_face_swapped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

      #new_face_swapped=Image.fromarray(new_face_swapped)
      #new_face_swapped.save('face_makeup.png')
    if mode>0:
      cap.release()
    if mode==1:
      outwriter.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = setup_argparser()

    parser.add_argument(
        "--mode",
        default=0,
        help="0 for image, 1 for video and 2 for stream")
    parser.add_argument(
        "--inputpath",
        default='bushra.jpg',
        help="Input path of image or video")
    parser.add_argument(
        "--reference_dir",
        default="assets/images/makeup",
        help="path to reference images")
    parser.add_argument(
        "--useseg",
        default=1,
        help="Use face segmentation")
    parser.add_argument(
        "--device",
        default="cuda",
        help="device used for inference")
    parser.add_argument(
        "--model_path",
        default="assets/models/G.pth",
        help="model for loading")

    args = parser.parse_args()
    args.useseg=int(args.useseg)
    args.inputpath='../tests/'+args.inputpath
    mode=int(args.mode)
    evaluate(args,mode=mode, cp='faceparsing/79999_iter.pth')


