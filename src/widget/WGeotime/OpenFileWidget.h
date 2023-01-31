#ifndef __OPENFILEWIDGET__H__
#define __OPENFILEWIDGET__H__

#include <QDialog>
#include <QListWidget>

#include "seismic3dabstractdataset.h"

class QComboBox;
class SeismicSurvey;
class Seismic3DAbstractDataset;
class WorkingSetManager;

class OpenFileWidget: public QDialog {
Q_OBJECT
public:
	OpenFileWidget(QWidget *parent, std::vector<QString> vTinyNames, std::vector<QString> vFullNames, bool multiSelection = false);
	virtual ~OpenFileWidget();
	QString getSelectedTinyName();
	QString getSelectedFullName();

private:
	QListWidget *listFile;
	QString tinyName;
	QString fullName;
	std::vector<QString> vTinyNames;
	std::vector<QString> vFullNames;
	bool multiSelection;
	int getIndexFromVectorString(std::vector<QString> list, QString txt);
	QString getFullNameFromTinyName(QString tinyName);

private slots:
	void accepted();
	void listFileDoubleClick(QListWidgetItem* item);
};


#endif
