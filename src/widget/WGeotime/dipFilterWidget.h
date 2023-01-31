/*
 *
 *
 *  Created on: 11 September 2020
 *      Author: l1000501
 */

#ifndef __DIPFILTERWIDGET_
#define __DIPFILTERWIDGET__

#include <QThread>
#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QComboBox>
#include <QGroupBox>
#include <QDialog>
#include <QListWidget>


#include <vector>
#include <math.h>

#include "GeotimeSystemInfo.h"
#include <GeotimeConfiguratorWidget.h>
#include "ProjectManagerWidget.h"


class QTableView;
class QStandardItemModel;
class OrientationWidget;
class WorkingSetManager;

#ifndef MIN
    #define MIN(x,y)		( ( x >= y ) ? y : x )
#endif

#ifndef MAX
    #define MAX(x,y)		( ( x >= y ) ? x : y )
#endif




class DipFilterWidget : public QWidget{
	Q_OBJECT
public:
	DipFilterWidget(WorkingSetManager *workingSetManager, QWidget* parent = 0);
	virtual ~DipFilterWidget();
	void trt_start();
	int GLOBAL_RUN_TYPE;
	std::vector<QString> listQStringDialog;
	void setProjectManagerWidget(ProjectManagerWidget *p) { m_projectManager = p; }


private:
	QLabel *m_processing = nullptr;
	GeotimeSystemInfo *systemInfo = nullptr;
	QLineEdit *qlineedit_outfilename,
	*qlineedit_dx, *qlineedit_dy, *qlineedit_dz;
	QListWidget *lw_dataIn;
	QComboBox *qcombobox_type;
	QProgressBar *qpb_progress;
	QString seismic_name;
	std::vector<QString> v_seismicFilename;
	int getIndexFromVectorString(std::vector<QString> list, QString txt);
	void get_size_from_filename(QString filename, int *size);
	int parameters_check();
	QString seismicFilenameToOutFilename(QString seismicFilename, QString suffix);
	ProjectManagerWidget *m_projectManager = nullptr;
	OrientationWidget *m_orientationWidget = nullptr;
	void processingDisplay();
	unsigned int m_cptProcessing = 0;
	WorkingSetManager *m_workingSetManager = nullptr;
	bool checkDataSetsDims(std::vector<QString> paths);
	bool checkFormat(std::vector<QString> &path);
	int m_dimx = -1;
	int m_dimy = -1;
	int m_dimz = -1;

private slots:
	void trt_seismic_open();
	void trt_launch_thread();
	void trt_stop();
	void showTime();
	// void trt_session_load();
	// void trt_session_save();
	void trt_seismic_clear();
};

class MyThreadDipFilter : public QThread
{
     // Q_OBJECT
	 public:
	MyThreadDipFilter(DipFilterWidget *p);
	 private:
	DipFilterWidget *pp;

protected:
     void run();
};




#endif /* MURATAPP_SRC_TOOLS_XCOM_MARFACOMPUTATIONWIDGET_H_ */
