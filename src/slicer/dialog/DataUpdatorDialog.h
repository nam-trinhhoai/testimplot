#ifndef DataUpdatorDialog_H
#define DataUpdatorDialog_H

#include <QDialog>

#include "seismic3dabstractdataset.h"
#include <WellUtil.h>

class QComboBox;
class SeismicSurvey;
class Seismic3DAbstractDataset;
class ManagerUpdateWidget;
class WorkingSetManager;

class DataUpdatorDialog: public QDialog {
Q_OBJECT
public:
	DataUpdatorDialog(QString dataName,WorkingSetManager *manager,QWidget *parent);
	virtual ~DataUpdatorDialog();

	std::vector<QString> getPathSelected();

    bool forceAllItems() const;
    void setForceAllItems(bool val);

protected:
	ManagerUpdateWidget* m_Updator = nullptr;
	WorkingSetManager *m_manager;
	QString m_DataName;
private slots:
	void accepted();
};

#endif
