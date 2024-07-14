from turtle import width
import streamlit as st
import cv2
import numpy as np
import layoutparser as lp
from layoutparser.models import Detectron2LayoutModel
from layoutparser.visualization import draw_box
from io import BytesIO

# Visualize the detected layout
color_map = {
    'Title': 'red',
    'Text':'green',
    'List':'blue',
    'Table':'yellow',
    'Figure':'purple'
}

with st.sidebar:
    file = st.file_uploader("Upload PDF",   type=["PNG","JPEG"],accept_multiple_files=False)   
    threshold = st.slider("Set Threshold",min_value=0.0, max_value=1.0, step=0.05,value=0.8)
    if threshold:
        model = Detectron2LayoutModel(config_path='config/config.yaml',
                                model_path='models/model_final.pth',
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", threshold],
                                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = image[..., ::-1]
    #st.image(image, caption="Original Image", use_column_width=True)
    
    # Identify the layout on the image
    layout = model.detect(image)

    processed_image = draw_box(image,layout,box_width=3,color_map=color_map,show_element_type=True,show_element_id=True,)
    buf = BytesIO()
    processed_image.save(buf,format="JPEG")

    st.image(processed_image,caption="Processed Image", use_column_width=True)  

    st.download_button(
        label="Download Image",
        data=buf.getvalue(),
        file_name="downloaded_image.jpg",
        mime="image/jpeg"
    )


