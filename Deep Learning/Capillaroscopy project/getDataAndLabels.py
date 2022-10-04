# Function to retrieve data and labels
def getDataAndLabels(path,size):

  data_dir = '/content/gdrive/MyDrive/Semillero/RIM-ONE_DL_images/partitioned_randomly'
  files_dir = os.path.join(data_dir,path)
  files_data = len(os.listdir(files_dir))*[None]
  i = 0

  labels_dir = '/content/gdrive/MyDrive/Semillero/RIM-ONE_DL_reference_segmentations'
  lfiles_dir = os.path.join(labels_dir,path[path.index('/')+1:])
  labels_id = []

  # Create list with filenames for glaucoma and normal labels
  with os.scandir(lfiles_dir) as l_files:
    for file in l_files:
      if file.name.endswith('.png'):
        labels_id.append(file.name)

  labels_id.sort()
  
  # Create image dataset
  files_labels = len(os.listdir(files_dir))*[None]

  with os.scandir(files_dir) as files:
    for file in files:
      img_path = os.path.join(files_dir,file)
      img = cv2.imread(img_path,1)/255
      b,g,r = cv2.split(img)
      img = cv2.merge([r,g,b])
      img = img_to_array(img,data_format='channels_last')
      img = tf.image.resize_with_pad(img,size[0],size[1])
      files_data[i] = img

      for k in range(len(labels_id)):
        if file.name[0:8] == labels_id[k][0:8]:
          limg_path = os.path.join(lfiles_dir,labels_id[k])
          limg = mimg.imread(limg_path)
          limg = img_to_array(limg,data_format='channels_last')
          limg = tf.image.resize_with_pad(limg,388,388)
          limg = np.rint(limg).astype('uint8')
          files_labels[i] = limg
          labels_id.pop(k)
          labels_id.pop(k)
          break
      i += 1

  files_data = np.asarray(files_data)
  files_labels = np.asarray(files_labels)

  return files_data, files_labels
