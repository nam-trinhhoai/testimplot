
#ifndef __PATCHWIDGET__
#define __PATCHWIDGET__






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
class WorkingSetManager;


class PatchWidget : public QWidget{
	Q_OBJECT
public:
	PatchWidget(WorkingSetManager *workingSetManager, QWidget* parent = 0);
	virtual ~PatchWidget();

	bool getCompute();
	QString getPatchName();
	QString getPatchPath();
	int getPatchSize();
	bool getFaultMaskInput();
	QString getFaultMaskPath();
	int getFaultThreshold();
	QString getPatchPolarity();
	double getDeltaVOverV();
	int getGradMax();
	std::vector<QString> getHorizonPaths();
	bool getPatchEnable();
	void setConstraintsDims(int dimx, int dimy, int dimz);

private:
	QGroupBox* groupBox = nullptr;
	QCheckBox *checkBoxCompute = nullptr;
	QCheckBox *faultInput = nullptr;
	FileSelectWidget *m_patchFileSelectWidget = nullptr;
	FileSelectWidget *m_faultFileSelectWidget = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;
	ProjectManagerWidget *m_projectManager = nullptr;
	PatchParametersWidget *m_patchParameters = nullptr;
	QCheckBox *m_enablePatch = nullptr;
	int m_dimx = -1;
	int m_dimy = -1;
	int m_dimz = -1;

	private slots:
	void trt_setEnabled(bool val);
};


#endif
