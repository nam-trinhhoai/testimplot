/*
 * 
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */

#ifndef MURATAPP_SRC_TOOLS_XCOM_GEOTIMESYSTEMINFO_H_
#define MURATAPP_SRC_TOOLS_XCOM_GEOTIMESYSTEMINFO_H_

#include <vector>

#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QListWidget>
#include <QDir>
#include <QLineEdit>
#include <QTabWidget>
#include <QGroupBox>
#include <QProgressBar>


#include <vector>
#include <math.h>

// class QTableView;
// class QStandardItemModel;



class GeotimeSystemInfo : public QWidget{
	Q_OBJECT
public:
    GeotimeSystemInfo(QWidget* parent = 0);
	virtual ~GeotimeSystemInfo();
    void get_valid_gpu(int *tab, int *size);
    int get_gpu_nbre();
    double cuda_min_free_memory();
    static double qt_cpu_free_memory();
	
private:
    int nb_gpu, nb_threads;
    QGroupBox *qgb_systeminfo;
    QProgressBar **qpb_gpumemory, *qpb_cpumemory;
    QCheckBox **qcb_validgpu;
    QTimer *timer;
    QLabel *qlabel_nbthreads, *qlabel_cpumem, **qlabel_gpumem; 

    int get_valid_gpu_nbre();
    static double qt_cpu_total_memory();

private slots:
    void showTime();    
	


    // void trt_ok();
    // void trt_cancel();
	// void computeZvsRho();
};


#endif 
