'''
So I'm importing CV2 to I'm reading in that file.

And while this is true, I'm going to keep showing that puppy in its image and its new window, and

I'm going to at least wait one millisecond and then wait for the user to press the escape key before

I break out of this wild loop.

And then I'm going to destroy all the windows, essentially close everything.


'''

import cv2

img=cv2.imread('./DATA/00-puppy.jpg')

'''
Now if I were to just have that by itself, the problem is this will just continually show the image
forever over and over again.
So I need to have some sort of break condition to break out of this while true loop.
'''
while True:
    cv2.imshow('Puppy',img)

    #s is if We've waited at least one millisecond We've pressed the escape key.
    if cv2.waitKey(1) & 0xFF == 27:
        break;
cv2.destroyAllWindows()