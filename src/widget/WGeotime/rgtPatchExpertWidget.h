
#ifndef __RGTPATCHEXPERTWIDGET__
#define __RGTPATCHEXPERTWIDGET__

#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <vector>
#include <math.h>

class QTableView;
class QStandardItemModel;

#include <GeotimeConfiguratorWidget.h>



class RgtPatchExpertWidget : public QWidget{
	Q_OBJECT
public:
	RgtPatchExpertWidget(GeotimeConfigurationWidget *pconfig, QWidget* parent = 0);
	virtual ~RgtPatchExpertWidget();

private:
    QLineEdit *lineEditPatchSize, *lineEditPatchFitThreshold,
	*lineEditRgtIter, *lineeditTimeSmoothparameter,
	*lineEditPatchFaultMaskThreshold,
	*lineeditRgtDecimY, *lineeditRgtDecimZ,
	*lineeditPatchDeltaVoverV,
	*lineeditPatchGradMax,

	*lineEditRgtIterScaleInit, *lineeditTimeSmoothparameterScaleInit,
	*lineeditRgtDecimYScaleInit,
	*lineEditRgtIdleDipMax;

    QComboBox *cb_stackformat, *qcb_fileformat;
    QComboBox *cbPatchPolarity;
    QCheckBox *qcbScaleInit;
    QCheckBox *qcb_snapping, *qcb_seedthreshold_valid;
    QPushButton *pbSampleLimits;
    // QCheckBox *cb_partial_rgt_save;
    GeotimeConfigurationWidget *pconf;

    void fill_fields();


	// MarfaConfiguratorWidget* marfaConfigurator;
	// MarfaDataDebugModel* debugModel;

private slots:
    void trt_ok();
    void trt_cancel();
	// void computeZvsRho();
};


#endif /* MURATAPP_SRC_TOOLS_XCOM_MARFACOMPUTATIONWIDGET_H_ */
