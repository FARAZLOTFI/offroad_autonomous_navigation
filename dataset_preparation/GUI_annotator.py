import cv2
import os
import tkinter as tk 
from tkinter import filedialog
from PIL import ImageTk,Image
from tkinter import messagebox
import time

class image_anotator():
    def __init__(self):
        self.counter_frame = 0 
        self.flag_save = False
        self.image_label = None
        self.list_labels = ['Tree', 'Other_obstacles', 'Human', 'Waterhole', 'Mud', 'Jump', 'Traversable_grass',
                            'Smooth_road']

        self.annotator = tk.Tk()
        self.annotator.geometry("640x480")
        self.annotator.title('MRL Lab Annotation Software')
        self.menu = tk.Menu(self.annotator)
        self.annotator.config(menu=self.menu)
        self.filemenu = tk.Menu(self.menu)
        self.menu.add_cascade(label='File', menu=self.filemenu)
        self.filemenu.add_command(label='Open the image folder',command = self.open_images_folder)
        self.filemenu.add_command(label='Open the annotation folder',command = self.open_annotation_folder)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='Exit', command=self.annotator.quit)
        
        
        self.button1 = tk.Button(self.annotator, text='Previous', width=35, command = self.previousFrame)
        self.button1.pack()
        self.button1.place(x=400, y=850)
        
        self.button2 = tk.Button(self.annotator, text='Save', width=35, command = self.save_annotations)
        self.button2.pack()
        self.button2.place(x=660, y=850)
        
        self.button3 = tk.Button(self.annotator, text='Next', width=35, command = self.nextFrame)
        self.button3.pack()
        self.button3.place(x=930, y=850)
        
        #self.button4 = tk.Button(self.annotator, text='Negative sample', width=35, command = self.negative_sample)
        #self.button4.pack()
        #self.button4.place(x=1200, y=850)
        
        self.annotator.bind('<Motion>', self.motion)
        #(tree, other_obstacles, Human, waterhole, mud, jump, traversable_grass, smooth_road)
        self.annotator.bind('0',self.annot) # tree
        self.annotator.bind('1',self.annot) # other_obstacles
        self.annotator.bind('2',self.annot) # Human
        self.annotator.bind('3', self.annot)# waterhole
        self.annotator.bind('4', self.annot)# mud
        self.annotator.bind('5', self.annot)  # jump
        self.annotator.bind('6', self.annot)  # traversable_grass
        self.annotator.bind('7', self.annot)  # smooth_road

        self.annotator.bind('<space>',self.skip_labeled_images)#
        #self.annotator.bind('<Escape>',self.annot)#

        self.annotator.bind('n', self.nextFrame)  #
        self.annotator.bind('b', self.previousFrame)  #

        img = Image.open('./mrl.png')
        img = ImageTk.PhotoImage(img)
        self.panel = tk.Label(self.annotator, image=img)
        self.panel.image = img
        self.panel.pack()
        self.panel.place(x=50, y=100)

        # Create label
        l = tk.Label(self.annotator, text = "MRL lab annotation software for the classification task")
        l.config(font =("Calibri", 12,"bold"))
        l.pack()
        l.place(x=40, y=30)#y=310)

        # Create label
        l = tk.Label(self.annotator, text = "Guideline: Please follow the steps to annotate each image:"+
                                            "\n" +
        "\n"+"1- Determine the images' and the annotations' directories."+
                                            "\n"+
        "\n"+"2- Choose the image label from the following options:"+
                                            "\n"+
        "\n"+"Tree (0), Other_obstacles (1), Human (2), Waterhole (3), Mud (4), Jump (5)"+
                                            "\n"+
        "\n" +"Traversable_grass (6), Smooth_road (7)"
                                            "\n"+
        "\n"+"For each option press the corresponding number from your keyboard"+
                                            "\n"+
        "\n"+"Use 'n' and 'b' to go to the next/previous image"
                                            "\n"+
        "\n"+"Lastly, use 'space' to directly get to the unlabeled images"
                                           "\n" +
        "\n"+"Good luck :D ")

        l.config(font =("Time New Roman", 10))
        l.pack()
        l.place(x=1300, y=300)

        self.info_ = tk.Label(self.annotator, text = "Waiting for your command!")

        self.info_.config(font =("Time New Roman", 10))
        self.info_.pack()
        self.info_.place(x=660, y=880)

        self.list_images = []

        self.flag_annotated = False


    def open_images_folder(self):
        self.main_path = filedialog.askdirectory()+'/'
        self.list_images = os.listdir(self.main_path)
        self.list_images.sort()
    def open_annotation_folder(self):
        self.main_path_GT = filedialog.askdirectory()+'/'
        if(len(self.list_images)>0 and self.counter_frame == 0): # This is to handle the first image! 
            self.path_image = self.main_path + self.list_images[self.counter_frame]
            self.open_img()
            self.open_GT()

    def nextFrame(self,event=None):
        self.info_["text"] = "Not labeled yet!"
        
        if(self.flag_save or self.counter_frame == 0 or self.flag_annotated):
            self.image_label = None

            if self.counter_frame<(len(self.list_images) - 1):
                self.counter_frame = self.counter_frame + 1

                self.path_image = self.main_path + self.list_images[self.counter_frame]

                self.open_img()
                self.open_GT()

                self.flag_save = False

            else:
                self.info_["text"] = "You have annotated all the images!"



        else:
            messagebox.showinfo("Warning", "This image is not annotated yet!")
            self.flag_save = True

    def previousFrame(self,event=None):
        self.info_["text"] = "Not labeled yet!"
        
        if(self.flag_save or self.counter_frame == 0 or self.flag_annotated):
            self.image_label = None
                
            if not(self.counter_frame == 0):
                self.counter_frame = self.counter_frame - 1 
                
            self.path_image = self.main_path+self.list_images[self.counter_frame]

            self.open_img()
            self.open_GT()    
        
            self.flag_save = False
            
        else:
            messagebox.showinfo("Warning", "This image is not annotated yet!")
            self.flag_save = True

    def skip_labeled_images(self,event):
        self.counter_frame = len(os.listdir(self.main_path_GT)) - 2
    def motion(self,event):
        self.mouse_x, self.mouse_y = event.x, event.y
        #print('{}, {}'.format(mouse_x, mouse_y))

    def open_img(self):

        img_ = cv2.imread(self.path_image)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB )
        img_ = cv2.resize(img_,(800,800))
        img_ = cv2.rectangle(img_, (160,600), (550,800), color=(255, 0, 0),thickness=2)

        img_ = Image.fromarray(img_)
        img_ = ImageTk.PhotoImage(img_)
        self.panel = tk.Label(self.annotator, image=img_)
        self.panel.pack()
        self.panel.place(x=400, y=50)
        self.panel.image = img_

    def load_annotation(self, path):
        # perhaps txt files
        f = open(path, "r")
        lines = f.readlines()
        self.image_label = int(float(lines[0]))

    def open_GT(self):
        try:
            self.image_label = None
            #self.image_label = self.load_annotation(self.main_path_GT + self.list_images[self.counter_frame][:-4] + '_debug.txt')
            self.load_annotation(self.main_path_GT + self.list_images[self.counter_frame][:-4] + '.txt')
            self.info_["text"] = 'The current label is: '+ self.list_labels[self.image_label]
            time.sleep(1)
            ##self.annot(flag_recovery=True)
            self.flag_annotated = True
        except:
            self.image_label = None
            self.flag_annotated = False

    def annot(self,event=None,flag_recovery=False):
        if not(flag_recovery):
            key_ = event.keysym
        else:
            key_ = 'l'

        # img_ = cv2.imread(self.path_image)
        # img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB )
        # img_ = cv2.resize(img_,(800,800))
        
        #(tree, other_obstacles, human, waterhole, mud, jump, traversable_grass, smooth_road)
        if(key_ == '0'):
            self.info_["text"] = 'Labeled as Tree'
            self.image_label = 0
        elif(key_ == '1'):
            self.info_["text"] = 'Labeled as Obstacle'
            self.image_label = 1
        elif(key_ == '2'):
            self.info_["text"] = 'Labeled as Human'
            self.image_label = 2
        elif(key_ == '3'):
            self.info_["text"] = 'Labeled as Waterhole'
            self.image_label = 3
        elif(key_ == '4'):
            self.info_["text"] = 'Labeled as Mud'
            self.image_label = 4
        elif(key_ == '5'):
            self.info_["text"] = 'Labeled as Jump'
            self.image_label = 5
        elif(key_ == '6'):
            self.info_["text"] = 'Labeled as Traversable grass'
            self.image_label = 6
        elif(key_ == '7'):
            self.info_["text"] = 'Labeled as Smooth road'
            self.image_label = 7

        self.save_annotations()
        self.image_label = None

        # img_ = Image.fromarray(img_)
        # img_ = ImageTk.PhotoImage(img_)
        # self.panel = tk.Label(self.annotator, image=img_)
        # self.panel.pack()
        # self.panel.place(x=400, y=50)
        # self.panel.image = img_

    def save_annotations(self,):
        output_size = (800,800,3)    
        model_input_size = (320,320)
        debug_mode = False
        if(self.image_label is not None):
            if not(debug_mode):
                with open(self.main_path_GT+self.list_images[self.counter_frame][:-4] + '.txt', 'w') as f:
                    f.write(str(self.image_label))
                f.close()
                time.sleep(0.3)

                #print('saved!')  
                self.info_["text"] = 'The ground truth has been generated; The image has been labeled as: ' + self.list_labels[self.image_label]
                self.flag_save = True
        else:
            messagebox.showinfo("Error!", "Some thing went wrong; perhaps you missed sth!")

    def negative_sample(self):
        self.image_label = -1
        self.save_annotations()

if __name__ == "__main__":
    
    GUI_instance = image_anotator() 


    GUI_instance.annotator.mainloop()
