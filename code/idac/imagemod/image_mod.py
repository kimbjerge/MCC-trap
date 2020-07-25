import cv2

class Imagemod:
    def __init__(self):
        print('image_mod ready')

    def drawoois(self, image, ois):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for oi in ois:
            cv2.rectangle(image, (oi.x, oi.y), (oi.x + oi.w, oi.y + oi.h), (0, 255, 0), 2)
            if oi.y < 30:
                cv2.putText(image, 'id: ' + str(oi.id) + ' Class: ' + oi.label, (oi.x, oi.y + 30 + oi.h), font, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Percent : %.2f%%' % oi.percent, (oi.x, oi.y+60+oi.h), font, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(image, 'id: ' + str(oi.id) + ' Class: ' + oi.label, (oi.x, oi.y-30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Percent : %.2f%%' % oi.percent, (oi.x, oi.y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if len(oi.centerhist) > 3:
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 1], oi.centerhist[len(oi.centerhist) - 2],
                         (0, 255, 0), 8)
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 2], oi.centerhist[len(oi.centerhist) - 3],
                         (0, 255, 0), 8)
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 3], oi.centerhist[len(oi.centerhist) - 4],
                         (0, 255, 0), 8)

            elif len(oi.centerhist) > 2:
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 1], oi.centerhist[len(oi.centerhist) - 2],
                         (0, 255, 0), 8)
                cv2.line(image, oi.centerhist[len(oi.centerhist) - 2], oi.centerhist[len(oi.centerhist) - 3],
                         (0, 255, 0), 8)
            elif len(oi.centerhist) > 1:
                cv2.line(image, oi.centerhist[len(oi.centerhist)-1], oi.centerhist[len(oi.centerhist)-2], (0, 255, 0), 8)
        return image
