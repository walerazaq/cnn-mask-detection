
# CNN for Detecting and Determining Proper Mask Usage

This project implements a Convolutional Neural Network (CNN) to detect and classify the proper usage of face masks in images. The model can identify the following classes:

- **`without_mask`**: No mask is worn.
- **`with_mask`**: Mask is properly worn.
- **`mask_weared_incorrect`**: Mask is worn incorrectly.

### Possible Use Cases

1. **Public Spaces Monitoring**: Automated mask detection in airports, train stations, malls, or other high-traffic areas to ensure compliance with health guidelines.
2. **Healthcare Facilities**: Ensuring proper mask usage in hospitals and clinics to maintain a safe environment.
3. **Educational Institutions**: Monitoring mask compliance in schools, colleges, and universities.
4. **Workplace Safety**: Enhancing workplace safety by verifying mask usage in industries and offices.
5. **Smart City Applications**: Integrating mask detection into smart surveillance systems for urban health monitoring.

## Features

- **Data Preparation**: Splits the dataset into training, validation, and testing subsets.
- **Transformations**: Applies advanced data augmentation techniques using the `albumentations` library to improve model robustness and generalisation.
- **Model**: Utilises Faster R-CNN with a ResNet-50 backbone for object detection, fine-tuned for the task.
- **Visualisation**: Includes functionality to visualise bounding boxes and classifications on test images.

## Sample Predictions

Below are sample predictions showcasing the model's performance:
![__results___21_5](https://github.com/user-attachments/assets/0348e8e3-f586-45ef-85e7-c45ed68255e0)
![__results___21_3](https://github.com/user-attachments/assets/c243d977-4365-4a4f-bf96-0fbbf85bd872)
![__results___21_1](https://github.com/user-attachments/assets/3bf58feb-bce2-4fd3-8f02-214ae243afc7)
![__results___21_9](https://github.com/user-attachments/assets/434ab2b8-4f08-4879-81b1-d0a33e4038ac)
![__results___21_7](https://github.com/user-attachments/assets/f6ffa853-c1e0-4448-b800-6d3c77e053e6)
