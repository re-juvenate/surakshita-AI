# import requests
# from app import db

# def send_image(uid):
#     url = "https://example.com/api"
#     headers = {"Content-Type": "multipart/form-data"}
#     data = {"uid": uid}

#     files = {"image": db.get_image(uid)}
#     response = requests.post(url, headers=headers, data=data, files=files)

#     if response.status_code == 200:
#         response_data = response.json()
#         return response_data
#     else:
#         return None

# def send_image_with_pii(id, pii):
#     url = "https://example.com/api"
#     headers = {"Content-Type": "multipart/form-data"}
#     data = {"id": id, "pii": pii}

#     files = {"image": db.get_image(id)}
#     response = requests.post(url, headers=headers, data=data, files=files)

#     if response.status_code == 200:
#         response_data = response.json()
#         return response_data
#     else:
#         return None

