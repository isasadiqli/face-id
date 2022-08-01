import os
import threading

import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen, SlideTransition
import uuid

import face_recognition


class OpenCameraScreen(Screen):
    def __init__(self, **kwargs):
        super(OpenCameraScreen, self).__init__(**kwargs)

        self.count_next = 0
        self.label_text = 'Taking photos for positives'

        self.webcam = Image(size_hint=(1, .6))
        self.back = Button(text='Back', size_hint=(1, .1), on_release=self.back_callback)
        self.start = Button(text='Start', on_release=self.start_callback)
        self.next = Button(text='Next', disabled=True, on_release=self.next_callback)
        self.label = Label(text=self.label_text, size_hint=(1, 0.1))

        self.progress_bar = ProgressBar(max=300, size_hint=(1, 0.1))

        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        button_layout.add_widget(self.start)
        button_layout.add_widget(self.next)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.back)
        layout.add_widget(self.webcam)
        layout.add_widget(self.label)
        layout.add_widget(self.progress_bar)
        layout.add_widget(button_layout)

        self.add_widget(layout)

    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture

    def back_callback(self, instance):
        Clock.unschedule(self.update)
        self.capture.release()
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'train_screen'

    def start_callback(self, instance):
        path = 'anchor' if self.count_next == 1 else 'positive'

        # import os
        # import glob
        #
        # files = glob.glob(os.path.join('data', path, '*'))
        # for f in files:
        #     os.remove(f)

        t = threading.Thread(target=face_recognition.take_images,
                             args=(path, self.capture, self.next, self.progress_bar))
        t.start()
        # self.take_images()

    def next_callback(self, instance):
        self.count_next += 1

        self.next.disabled = True

        self.label.text = 'Taking photos for anchor'

        if self.count_next == 2:
            Clock.unschedule(self.update)
            self.capture.release()
            self.manager.transition = SlideTransition(direction='right')

            self.manager.current = 'training_information_screen'
            i = face_recognition.find_screen(self.manager.children, 'training_information_screen')

            t = threading.Thread(target=face_recognition.run_train, args=(self.manager.children[i],))

            t.start()
