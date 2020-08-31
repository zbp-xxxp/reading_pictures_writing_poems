import paddlehub as hub

if __name__ == "__main__":
    readingPicturesWritingPoems = hub.Module(directory="./reading_pictures_writing_poems")
    readingPicturesWritingPoems.WritingPoem(image = "scenery.jpg", use_gpu=True)