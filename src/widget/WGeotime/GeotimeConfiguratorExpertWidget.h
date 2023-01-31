/*
 *
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */

#ifndef MURATAPP_SRC_TOOLS_XCOM_GEOTIMECONFIGURATIONEXPERTWIDGET_H_
#define MURATAPP_SRC_TOOLS_XCOM_GEOTIMECONFIGURATIONEXPERTWIDGET_H_

#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <vector>
#include <math.h>
#include "GeotimeConfiguratorWidget.h"

class QTableView;
class QStandardItemModel;




class GeotimeConfigurationExpertWidget : public QWidget{
	Q_OBJECT
public:
	GeotimeConfigurationExpertWidget(GeotimeConfigurationWidget *pconfig, QWidget* parent = 0);
	virtual ~GeotimeConfigurationExpertWidget();

private:
    QLineEdit *lineedit_iteration, *lineedit_dipthreshold, *lineedit_decimationfactor, /**lineedit_rgtsaverate, */
    *lineedit_sigmagradient, *lineedit_sigmatensor, *lineedit_rgtcompresserror, *lineedit_nbthreads,
	*lineedit_seedthreshold;
    QComboBox *cb_stackformat, *qcb_fileformat;
    QCheckBox *qcb_snapping, *qcb_seedthreshold_valid;
    QPushButton *pbSampleLimits;
    // QCheckBox *cb_partial_rgt_save;
    GeotimeConfigurationWidget *pconf;

    void fill_fields();
    bool checkEnableButtonSampleLimits();


	// MarfaConfiguratorWidget* marfaConfigurator;
	// MarfaDataDebugModel* debugModel;

private slots:
    void trt_ok();
    void trt_cancel();
    void trt_seedThresholdValid(bool val);
    void trt_sampleLimits();
	// void computeZvsRho();
};


#endif /* MURATAPP_SRC_TOOLS_XCOM_MARFACOMPUTATIONWIDGET_H_ */
