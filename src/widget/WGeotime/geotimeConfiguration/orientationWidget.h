/*
 *
 *
 *  Created on:
 *      Author: l1000501
 */

#ifndef __ORIENTATIONWIDGET_H_
#define __ORIENTATIONWIDGET_H_

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


#include <vector>
#include <math.h>

class QTableView;
class QStandardItemModel;
class FileSelectWidget;
class ProjectManagerWidget;
class OrientationParametersWidget;
class WorkingSetManager;



class OrientationWidget : public QWidget{
	Q_OBJECT
public:
	OrientationWidget(ProjectManagerWidget *projectManager, bool enableParam = false, WorkingSetManager *workingSetManager = nullptr, QWidget* parent = 0);
	virtual ~OrientationWidget();
	QString getDipxyFilename();
	QString getDipxyPath();
	QString getDipxzFilename();
	QString getDipxzPath();
	int getProcessingTypeIndex();
	bool getComputationChecked();

	void setGradient(double val);
	void setTensor(double val);
	double getGradient();
	double getTensor();
	void setConstraintsDims(int dimx, int dimy, int dimz);




private:
	QGroupBox* groupBox = nullptr;
	QCheckBox *checkBoxCompute;
	QComboBox *comboProcessingType;
	FileSelectWidget *m_dipxyFileSelectWidget = nullptr;
	FileSelectWidget *m_dipxzFileSelectWidget = nullptr;
	ProjectManagerWidget *m_projectManager = nullptr;
	OrientationParametersWidget *m_parameters = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;
	int m_dimx0 = -1;
	int m_dimy0 = -1;
	int m_dimz0 = -1;

	private slots:
	void trt_setEnabled(bool val);

};




#endif
