
#ifndef __RGTPATCHMANAGERWIDGET__
#define __RGTPATCHMANAGERWIDGET__




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
#include <QTableWidget>
#include <QPushButton>
#include <QVBoxLayout>

#include <utility>
#include <WellUtil.h>
#include <ObjectManager.h>
// #include <ProjectManagerNames.h>

class RgtPatchManagerWidget : public ObjectManager{

public:
	RgtPatchManagerWidget(QWidget* parent = 0);
	virtual ~RgtPatchManagerWidget();

private:
	QLineEdit *qleRgtPatchPatchSize, *qleRgtPatchPatchName;

};







#endif
