from werkzeug.utils import secure_filename


def save_file(file):
    # Save the file securely
    filename = secure_filename(file.filename)
    file.save(filename)
    return filename
