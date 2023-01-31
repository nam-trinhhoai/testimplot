#ifndef __IMPORTSISMAGEHORIZONDIALOG__
#define __IMPORTSISMAGEHORIZONDIALOG__

#include <QDialog>
#include <QList>
#include <QString>

class WorkingSetManager;
class SeismicSurvey;
class Seismic3DAbstractDataset;

class QComboBox;
class QSpinBox;
class QGridLayout;

class ImportSismageHorizonWithWorkingSetWidget ;

class ImportSismageHorizonDialog : public QDialog {
	Q_OBJECT
public:
	ImportSismageHorizonDialog(WorkingSetManager* workingSet);
	// ImportSismageHorizonDialog(SeismicSurvey* survey);
	~ImportSismageHorizonDialog();

private:
	ImportSismageHorizonWithWorkingSetWidget *m_importSismageHorizonWithWorkingSetManager = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;

};

#endif
