# NBU-WLFI database

Benchmark Database and Objective Metric for Wide Field of View Light Field Image Quality Measurement
Rui Zhou, Gangyi Jiang, Yeyao Chen, Yueli Cui, Ting Luo, and Haiyong Xu.

The manuscript has been submitted to IEEE TIM.

### Get access to the NBU-WLFI database
Please download the full database on [Baidu drive](https://pan.baidu.com/s/1m5NRPBh-D6_Oy4OtX7HlDw) (code: NBU).

The database contains 10 reference wide field of view light field images with 20 distorted versions for each, and their corresponding subjective quality scores.

### Illustration of the central view of SRCs in the proposed database.
![image](https://github.com/oldblackfish/NBU-WLFI/blob/main/fig/SRCs.png)

The structure of the database is shown as follows:

    ------ NBU-WLFI
          ------ Subjective-WLFIQA
          ------ Distorted
                  -------- I01C01
                  -------- I01C02
                  -------- ...
                  -------- I01C10
                  -------- I02C01
                  -------- ...
                  -------- I10C01
                  -------- ...
                  -------- I10C20
          ------ Reference
                  -------- I01C00
                  -------- I02C00
                  -------- ...
                  -------- I10C00


# EPPVS-BWLFQ metric

### Train
Train the model using the following command:
```
python Train.py  --trainset_dir ./Datasets/NBU_WLFI_5x32x64/
```

### Test Overall Performance
Test the overall performance using the following command:
```
python Test.py
```

### Test Individual Distortion Type Performance
Test the individual distortion type performance using the following command:
```
 python Test_Dist.py
```

### Result
The performances of our EPPVS-BWLFQ metric on the NBU-WLFI datasets can reproduce these performances using the h5 results we provide in './EPPVS-BWLFQ/Results/...'.

### Contact
For any questions, feel free to contact: 2211100079@nbu.edu.cn or zhourui0628@163.com
