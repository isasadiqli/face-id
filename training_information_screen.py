from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen

import face_recognition


class TrainingInformationScreen(Screen):
    def __init__(self, **kwargs):
        super(TrainingInformationScreen, self).__init__(**kwargs)

        self.data_augmentation_pos_pb = ProgressBar(max=2700)
        self.data_augmentation_pos_lb = Label(text='Data augmentation is being processed for positive images:',
                                              halign='left', size_hint=(None, None))
        self.data_augmentation_pos_lb.bind(texture_size=self.data_augmentation_pos_lb.setter('size'))
        dap_layout = BoxLayout(orientation='vertical')
        dap_layout.add_widget(self.data_augmentation_pos_lb)
        dap_layout.add_widget(self.data_augmentation_pos_pb)

        self.data_augmentation_anc_pb = ProgressBar(max=2700)
        self.data_augmentation_anc_lb = Label(text='Data augmentation is being processed for anchor images:',
                                              halign='left', size_hint=(None, None))
        self.data_augmentation_anc_lb.bind(texture_size=self.data_augmentation_anc_lb.setter('size'))
        daa_layout = BoxLayout(orientation='vertical')
        daa_layout.add_widget(self.data_augmentation_anc_lb)
        daa_layout.add_widget(self.data_augmentation_anc_pb)

        self.getting_images_pb = ProgressBar(max=3)
        self.getting_images_lb = Label(text='Getting images:',
                                       halign='left', size_hint=(None, None))
        self.getting_images_lb.bind(texture_size=self.getting_images_lb.setter('size'))
        gi_layout = BoxLayout(orientation='vertical')
        gi_layout.add_widget(self.getting_images_lb)
        gi_layout.add_widget(self.getting_images_pb)

        self.creating_labeled_ds_pb = ProgressBar(max=3)
        self.creating_labeled_ds_lb = Label(text='Creating labeled datasets:',
                                            halign='left', size_hint=(None, None))
        self.creating_labeled_ds_lb.bind(texture_size=self.creating_labeled_ds_lb.setter('size'))
        clds_layout = BoxLayout(orientation='vertical')
        clds_layout.add_widget(self.creating_labeled_ds_lb)
        clds_layout.add_widget(self.creating_labeled_ds_pb)

        self.build_dataset_pipeline_pb = ProgressBar(max=3)
        self.build_dataset_pipeline_lb = Label(text='Building data loader pipeline:',
                                               halign='left', size_hint=(None, None))
        self.build_dataset_pipeline_lb.bind(texture_size=self.build_dataset_pipeline_lb.setter('size'))
        bdsp_layout = BoxLayout(orientation='vertical')
        bdsp_layout.add_widget(self.build_dataset_pipeline_lb)
        bdsp_layout.add_widget(self.build_dataset_pipeline_pb)

        self.make_train_test_data_pb = ProgressBar(max=7)
        self.make_train_test_data_lb = Label(text='Preparing train and test data:',
                                             halign='left', size_hint=(None, None))
        self.make_train_test_data_lb.bind(texture_size=self.make_train_test_data_lb.setter('size'))
        mttd_layout = BoxLayout(orientation='vertical')
        mttd_layout.add_widget(self.make_train_test_data_lb)
        mttd_layout.add_widget(self.make_train_test_data_pb)

        self.make_model_pb = ProgressBar(max=8)
        self.make_model_lb = Label(text='Building model:',
                                   halign='left', size_hint=(None, None))
        self.make_model_lb.bind(texture_size=self.make_model_lb.setter('size'))
        mm_layout = BoxLayout(orientation='vertical')
        mm_layout.add_widget(self.make_model_lb)
        mm_layout.add_widget(self.make_model_pb)

        self.train_pb = ProgressBar(max=13150)
        self.train_lb = Label(text='Training the model:',
                              halign='left', size_hint=(None, None))
        self.train_lb.bind(texture_size=self.train_lb.setter('size'))
        t_layout = BoxLayout(orientation='vertical')
        t_layout.add_widget(self.train_lb)
        t_layout.add_widget(self.train_pb)

        self.save_button = Button(text='Save', disabled=True, on_release=self.save_callback)
        self.model = None

        layout = BoxLayout(orientation='vertical', spacing=50, padding=50)
        layout.add_widget(dap_layout)
        layout.add_widget(daa_layout)
        layout.add_widget(gi_layout)
        layout.add_widget(clds_layout)
        layout.add_widget(bdsp_layout)
        layout.add_widget(mttd_layout)
        layout.add_widget(mm_layout)
        layout.add_widget(t_layout)
        layout.add_widget(self.save_button)

        self.add_widget(layout)

    def save_callback(self, instance):
        face_recognition.save_the_model(self.model, 'siamesemodel')
        self.manager.current = 'main_screen'
