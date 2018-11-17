import os
import cv2

path = "./components"
savePath = "./data"
numbers = [
    [ 0 ],
    [ 1 ],
    [ 2 ],
    [ 3 ],
    [ 4 ],
    [ 5 ],
    [ 5, 1 ],
    [ 5, 2 ],
    [ 5, 3 ],
    [ 5, 4 ],
    [ 10 ],
    [ 10, 1 ],
    [ 10, 2 ],
    [ 10, 3 ],
    [ 10, 4 ],
    [ 15 ],
    [ 15, 1 ],
    [ 15, 2 ],
    [ 18 ],
    [ 15, 4 ],
    [ 20 ],
    [ 20, 1 ],
    [ 20, 2 ],
    [ 20, 3 ],
    [ 20, 4 ],
]
categories = map(str, [ "bracket", 0, 1, 2, 3, 4, 5, 10, 15, 18, 20 ])
components = {}

def power2Set(n = 20):
    out = []

    for i in range(n):
        for j in range(n):
            out.append([i, j])

    return out

twoCombos = power2Set()

# Super freaking custom
def power3Set():
    out = []

    for i in range(8):
        for j in range(7):
            for k in range(8):
                out.append([i, j, k])

    return out

threeCombos = power3Set()

def getCombos(n):
    if n == 1:
        return twoCombos
    else:
        return threeCombos

for category in categories:
    for dirpath, dnames, fnames in os.walk("{}/{}".format(path, category)):
        components[category] = []
        for file in fnames:
            imagePath = os.path.join(dirpath, file)
            image = cv2.imread(imagePath)
            components[category].append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

def compose(A, B, C=None):
    img = cv2.addWeighted(A, 0.5,  B, 0.5, 0)
    if C is not None:
        img = cv2.addWeighted(img, 0.5,  C, 0.5, 0)
    inv = cv2.bitwise_not(img)
    ret, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY_INV)

    return thresh

for number, digits in enumerate(numbers):
    combos = getCombos(len(digits))

    for i, wumbo in enumerate(combos):
        bracket = components["bracket"][wumbo[0]]
        first = components[str(digits[0])][wumbo[1]]

        if len(digits) == 2:
            second = components[str(digits[1])][wumbo[2]]
        else:
            second = None

        img = compose(bracket, first, second)
        #img = cv2.resize(img, (50, 38))

        cv2.imwrite("{}/{}/{}.png".format(savePath, number, i), img)
