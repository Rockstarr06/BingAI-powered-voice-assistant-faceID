#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:07:21 2024

@author: Pravesh
"""

# Import necessary libraries
import openai
import asyncio
import re
import boto3
import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
import numpy as np
import cv2
import face_recognition
import os
import time
from pywhispercpp.model import Model

# Util functions
def play_audio(audio_data):
    """
    Play the synthesized audio response.
    """
    audio = np.frombuffer(audio_data, dtype=np.int16)
    playback.play(pydub.AudioSegment(
        data=audio.tobytes(),
        sample_width=2,  # Assuming 16-bit audio
        frame_rate=16000,
        channels=1  # Mono audio
    ))

def get_cookies(url):
    """
    Retrieve browser cookies for Bing authentication.
    """
    try:
        import browser_cookie3
        cookies = []
        cj = browser_cookie3.edge(domain_name=url)
        for cookie in cj:
            cookies.append(cookie.__dict__)
        return cookies
    except Exception as e:
        print("Error fetching cookies:", e)
        return None

# ASR (Automatic Speech Recognition)
class ASR:
    def __init__(self, model_name="tiny.en"):
        self.model = Model(model_name)

    def transcribe(self, audio_data):
        """
        Transcribe audio data using Whisper.
        """
        try:
            result = self.model.transcribe(media=audio_data.flatten())
            return ''.join([segment.text for segment in result])
        except Exception as e:
            print("Error transcribing audio:", e)
            return None

# Speech Synthesizer using Amazon Polly
class PollySynthesizer:
    def __init__(self):
        self.polly = boto3.client('polly', region_name='eu-west-2')

    def synthesize_speech(self, text):
        """
        Convert text to speech using Amazon Polly.
        """
        try:
            response = self.polly.synthesize_speech(
                Text=text, 
                OutputFormat='pcm', 
                VoiceId='Arthur', 
                Engine='neural'
            )
            return response['AudioStream'].read()
        except Exception as e:
            print("Error synthesizing speech:", e)
            return None

# Listener to capture user input via microphone
class Listener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.asr = ASR()

    def listen_for_user_input(self):
        """
        Listen to the user's input via microphone.
        """
        print("Speak now...")
        try:
            with sr.Microphone(sample_rate=16000) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=30)
                audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16).astype(np.float32) / (2 ** 15)
                return self.asr.transcribe(audio_data)
        except sr.WaitTimeoutError:
            print("Timeout waiting for user input.")
        except Exception as e:
            print("Error listening to user input:", e)
        return None

# Face Recognition and Login System
class FaceLogin:
    def __init__(self, synthesizer, listener):
        self.synthesizer = synthesizer
        self.listener = listener
        self.templates_dir = "User_templates"
        os.makedirs(self.templates_dir, exist_ok=True)
        self.registered_users = self.load_registered_users()

    def load_registered_users(self):
        """
        Load registered users from templates.
        """
        return [file.split("_")[0] for file in os.listdir(self.templates_dir) if file.endswith(".npy")]

    def save_user_template(self, encoding, user_name):
        """
        Save a user's face template to file.
        """
        np.save(os.path.join(self.templates_dir, f"{user_name}_template.npy"), encoding)

    def create_template(self, camera, num_samples=10):
        """
        Capture face encodings to create a user template.
        """
        play_audio(self.synthesizer.synthesize_speech("Please look at the camera."))
        encodings = []
        while len(encodings) < num_samples:
            ret, frame = camera.read()
            locations = face_recognition.face_locations(frame)
            if len(locations) == 1:
                encodings.append(face_recognition.face_encodings(frame, locations)[0])
        return np.mean(encodings, axis=0)

    def enroll_user(self):
        """
        Enroll a new user by capturing their face data.
        """
        user_name = self.ask_user_name()
        with cv2.VideoCapture(0) as camera:
            template = self.create_template(camera)
        self.save_user_template(template, user_name)
        play_audio(self.synthesizer.synthesize_speech(f"User {user_name} registered successfully."))

    def recognize_user(self):
        """
        Attempt to recognize a user based on face data.
        """
        play_audio(self.synthesizer.synthesize_speech("Looking for a match..."))
        with cv2.VideoCapture(0) as camera:
            end_time = time.time() + 5
            while time.time() < end_time:
                ret, frame = camera.read()
                locations = face_recognition.face_locations(frame)
                encodings = face_recognition.face_encodings(frame, locations)
                for encoding in encodings:
                    for user in self.registered_users:
                        template_path = os.path.join(self.templates_dir, f"{user}_template.npy")
                        if os.path.exists(template_path):
                            template = np.load(template_path)
                            if face_recognition.compare_faces([template], encoding)[0]:
                                play_audio(self.synthesizer.synthesize_speech(f"Welcome, {user}!"))
                                return user
        return None

    def ask_user_name(self):
        """
        Ask the user for their preferred name during enrollment.
        """
        play_audio(self.synthesizer.synthesize_speech("What is your name?"))
        return ''.join(filter(str.isalnum, self.listener.listen_for_user_input() or ""))

# Bing AI Assistant
class AIAssistant:
    def __init__(self, synthesizer, listener):
        self.cookies = get_cookies('.bing.com')
        self.synthesizer = synthesizer
        self.listener = listener

    async def query_bing(self, user_input):
        """
        Query Bing Chatbot for a response.
        """
        bot = Chatbot(cookies=self.cookies)
        try:
            response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)
            for msg in response["item"]["messages"]:
                if msg["author"] == "bot" and 'messageType' not in msg:
                    return re.sub(r'\[\^\d+\^\]', '', msg["text"])
        finally:
            await bot.close()

    async def run(self):
        """
        Main loop for the AI Assistant.
        """
        while True:
            play_audio(self.synthesizer.synthesize_speech("What can I help you with?"))
            user_input = self.listener.listen_for_user_input()
            if user_input:
                bot_response = await self.query_bing(user_input)
                play_audio(self.synthesizer.synthesize_speech(bot_response or "I'm sorry, I couldn't process that."))

# Main Functionality
if __name__ == "__main__":
    synthesizer = PollySynthesizer()
    listener = Listener()
    face_login = FaceLogin(synthesizer, listener)

    if face_login.recognize_user():
        assistant = AIAssistant(synthesizer, listener)
        asyncio.run(assistant.run())
    else:
        play_audio(synthesizer.synthesize_speech("Access denied."))
