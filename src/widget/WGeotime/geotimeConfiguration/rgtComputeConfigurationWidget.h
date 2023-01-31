
#ifndef __RGTCOMPUTECONFIGURATIONWIDGET__
#define __RGTCOMPUTECONFIGURATIONWIDGET__



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


class RgtComputeConfigurationWidget;



class RgtComputeConfigurationWidget : public QWidget{
	Q_OBJECT
public:
	RgtComputeConfigurationWidget(QWidget* parent = 0);
	virtual ~RgtComputeConfigurationWidget();

private slots:
};

#endif
