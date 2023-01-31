/*
 * 
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */

#ifndef MURATAPP_SRC_TOOLS_XCOM_GRADIENTMULTISCALEWIDGET_H_
#define MURATAPP_SRC_TOOLS_XCOM_GRADIENTMULTISCALEWIDGET_H_

#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <vector>
#include <math.h>

class QTableView;
class QStandardItemModel;


#ifndef MIN
    #define MIN(x,y)		( ( x >= y ) ? y : x )
#endif

#ifndef MAX
    #define MAX(x,y)		( ( x >= y ) ? x : y )
#endif


// class GeotimeConfigurationWidget;


class GradientMultiscaleWidget : public QWidget{
	Q_OBJECT
public:
	GradientMultiscaleWidget(QWidget* parent = 0);
	virtual ~GradientMultiscaleWidget();
	
private:
	

private slots:
	
	// void computeZvsRho();
};


#endif /* MURATAPP_SRC_TOOLS_XCOM_MARFACOMPUTATIONWIDGET_H_ */
