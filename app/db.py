import sqlite3
import uuid

# Connect to the database
conn = sqlite3.connect('image_db.sqlite')

# Create a cursor object
cursor = conn.cursor()

# Create a table to store images
cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uid TEXT,
        img BLOB,
        pii TEXT
    )
''')

# Function to add an image to the database
def add_image(img):
    image_uid = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO images (uid, img)
        VALUES (?, ?)
    ''', (image_uid, img))

    conn.commit()

# Function to retrieve an image based on its UID
def get_image(uid):
    cursor.execute('''
        SELECT data FROM images WHERE uid = ?
    ''', (uid,))
    result = cursor.fetchone()

    if result:
        return result[0]
    else:
        return None

# Function to add PII to the database based on the ID
def add_pii(uid, pii):
    cursor.execute('''
        UPDATE images SET pii = ? WHERE uid = ?
    ''', (pii, uid))
    conn.commit()

# Function to get PII based on UID
def get_pii(uid):
    cursor.execute('''
        SELECT pii FROM images WHERE uid = ?
    ''', (uid,))
    result = cursor.fetchone()

    if result:
        return result[0]
    else:
        return None

# Example usage
# image_data = b"binary_data_here"
# add_image(image_data)

# # Retrieve the image based on its UID
# image_uid = "your_image_uid_here"
# retrieved_image = get_image(image_uid)
# if retrieved_image:
#     print("Image found!")
#     # Do something with the retrieved image data
# else:
#     print("Image not found!")

# # Close the connection
# conn.close()
