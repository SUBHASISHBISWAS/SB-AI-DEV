import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

# cap.set(3, 300)
# cap.set(4, 300)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        font = cv2.FONT_HERSHEY_COMPLEX
        text = 'Width : ' + str(cap.get(3)) + ' Height : ' + str(cap.get(4))
        frame = cv2.putText(frame, text, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
