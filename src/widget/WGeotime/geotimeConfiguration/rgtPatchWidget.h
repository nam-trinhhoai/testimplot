
#ifndef __RGTPATCHWIDGET__
#define __RGTPATCHWIDGET__






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
class PatchParametersWidget;
class RgtPatchParametersWidget;
class ProcessRelay;
class WorkingSetManager;


class RgtPatchWidget : public QWidget{
	Q_OBJECT
public:
	RgtPatchWidget(WorkingSetManager *workingSetManager, QWidget* parent = 0);
	virtual ~RgtPatchWidget();
	bool getCompute();
	QString getRgtSuffix();

	int getScaleInitIter();
	double getScaleInitEpsilon();
	int getScaleInitDecim();
	int getIter();
	double getEpsilon();
	int getDecim();
	bool getScaleInitValid();
	void setDecim(int val);
	bool getIsRgtInit();
	QString getRgtInit();

	void setProcessRelay(ProcessRelay* relay);
	void setConstraintsDims(int dimx, int dimy, int dimz);


private:

	// QGroupBox* groupBox = nullptr;
	QCheckBox *m_patchRgtCompute = nullptr,
	*m_rgtVolumicRgt0 = nullptr;
	// QCheckBox *faultInput = nullptr;
	FileSelectWidget *m_rgtInitSelectWidget = nullptr;
	// FileSelectWidget *m_faultFileSelectWidget = nullptr;
	// ProjectManagerWidget *m_projectManager = nullptr;
	// PatchParametersWidget *m_patchParameters = nullptr;
	QLineEdit *m_patchRgtRgtName = nullptr;
	RgtPatchParametersWidget *m_patchParameters = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;
	int m_dimx = -1;
	int m_dimy = -1;
	int m_dimz = -1;

};


#endif
