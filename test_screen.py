import threading

import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import SlideTransition, Screen

import face_recognition


class TestScreen(Screen):
    def __init__(self, **kwargs):
        super(TestScreen, self).__init__(**kwargs)

        self.model = None

        self.webcam = Image(size_hint=(1, .8))
        self.verify = Button(text='Verify', size_hint=(1, .1), on_release=self.verify_callback)
        self.status = Label(text='Verification Uninitiated', size_hint=(1, .1))

        self.back = Button(text='Back', size_hint=(1, .1), on_release=self.back_callback)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.back)
        layout.add_widget(self.webcam)
        layout.add_widget(self.verify)
        layout.add_widget(self.status)

        self.add_widget(layout)

    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture

    def verify_callback(self, instance):
        t = threading.Thread(target=face_recognition.verify, args=(self.model, 0.5, 0.5, self.capture, self.status))
        t.start()

    def back_callback(self, instance):
        Clock.unschedule(self.update)
        self.capture.release()
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'main_screen'
