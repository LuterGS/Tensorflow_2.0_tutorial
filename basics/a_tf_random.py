import tensorflow as tf

if __name__ == "__main__":
    rand_uniform = tf.random.uniform([1],0,1)

    """
        random 에서, 균일분포 (uniform distribution) 함수에서 하나를 추출"
        1x1 행렬에서, 최소값 0, 최대값 1 사이의 값을 추출'
        일반적인 random함수같은 느낌. 최소값~최대값에서 가중치 없이 normal하게 추출'
    """
    rand_uniform2 = tf.random.uniform([1],-5,30)

    print(rand_uniform, rand_uniform2)



    rand_normal = tf.random.normal([3,2], 0, 1)
    sizeof_rand_normal = tf.rank(rand_normal)
    test = sizeof_rand_normal + 3
    print("sizeof rand ", sizeof_rand_normal)
    print("test : ", test)
    """
        random 에서, 정규분포 (normal dustribution) 함수에서 6개 (3x2 행렬이므로) 추출'
        3x2행렬에서 평균 0, 표준편차 1인 정규분포 추출'
        알던 정규분포 함수에서 추출한다고 하면 맞을거같음'
    """


    print(rand_normal)

    """    
        random함수 안의 데이터는 모두 float32로 저장되나보네'
    """