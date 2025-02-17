def compare_images_by_pixels(file1, file2):
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    
    # 이미지 크기 비교
    if image1.size != image2.size:
        print("이미지 형태(사이즈)가 다르네요")
        return False
    
    # 이미지 픽셀 데이터 비교
    pixels1 = np.array(image1)
    pixels2 = np.array(image2)
    return np.array_equal(pixels1, pixels2)



if __name__ == '__main__':
    # 예시 사용
    file1 = 'data/Spick_test_folder/record_renamed/1/AD/r10a01_1_10db_inc.png'
    file2 = 'data/Spick_test_folder/record_renamed/1/AD/r10a01_1_10db_reduce.png'
    are_same = compare_images_by_pixels(file1, file2)
    print("Images are the same by pixels:", are_same)
