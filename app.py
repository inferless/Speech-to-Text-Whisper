from faster_whisper import WhisperModel
import base64

class InferlessPythonModel:
        
    def initialize(self):
        model_size = "large-v3"
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    def base64_to_mp3(base64_data, output_file_path):
        mp3_data = base64.b64decode(base64_data)
        with open(output_file_path, "wb") as mp3_file:
                mp3_file.write(mp3_data)

    def infer(self, inputs):
        audio_data = inputs["audio_base64"]
        audio_file = "output.mp3"
        base64_to_mp3(audio_data,audio_file)
        segments, info = self.model.transcribe(audio_file, beam_size=5)
        text = ''.join([segment.text for segment in segments])
        
        return {"transcribed_output":text}

    def finalize(self):
        pass
