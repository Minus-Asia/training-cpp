import cv2


def draw_grid(img, num_column=6, num_row=5, top_dis=100, bottom_dis=100, left_dis=100, right_dis=100,
              line_color=(0, 255, 0), thickness=3, type_=cv2.LINE_AA):
    print(img.shape)
    img_grid = img.copy()

    grid_width = img.shape[1] - left_dis - right_dis
    grid_height = img.shape[0] - top_dis - bottom_dis

    step_row = int(grid_height / num_row)
    step_col = int(grid_width / num_column)
    x = left_dis
    y = top_dis
    while x < img_grid.shape[1] - right_dis:
        cv2.line(img_grid, (x, y), (x, img_grid.shape[0] - bottom_dis), color=line_color, lineType=type_, thickness=thickness)
        x += step_col

    x = left_dis
    while y < img_grid.shape[0] - bottom_dis:
        cv2.line(img_grid, (x, y), (img_grid.shape[1] - right_dis, y), color=line_color, lineType=type_, thickness=thickness)
        y += step_row

    scale = cv2.resize(img_grid, (0, 0), fx=0.2, fy=0.2)
    cv2.imshow("abc", scale)
    cv2.waitKey(0)


def crop_by_grid(img, num_column=6, num_row=5, top_dis=100, bottom_dis=100, left_dis=100, right_dis=100):
    img_list = []
    grid_width = img.shape[1] - left_dis - right_dis
    grid_height = img.shape[0] - top_dis - bottom_dis

    step_row = int(grid_height / num_row)
    step_col = int(grid_width / num_column)
    x = left_dis
    y = top_dis
    while x < img.shape[1] - right_dis and x + step_col < img.shape[1]:
        while y < img.shape[0] - bottom_dis and y + step_row < img.shape[0]:
            peace_img = img[y:y + step_row, x:x + step_col]
            img_list.append(peace_img)
            # cv2.imshow("111", peace_img)
            # cv2.waitKey(0)
            y += step_row
        y = top_dis
        x += step_col

    return img_list


if __name__ == '__main__':
    img = cv2.imread("/opt/MVS/bin/Temp/Data/Image_20220114135104731.bmp")
    draw_grid(img)
    list_img = crop_by_grid(img)

