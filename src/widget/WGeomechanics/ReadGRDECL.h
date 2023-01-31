/*
 *
 *
 *  Created on: 30 May 2022
 *      Author: l0359127
 */


#ifndef NEXTVISION_SRC_WIDGET_WReadGRDECL_ReadGRDECL_H_
#define NEXTVISION_SRC_WIDGET_WReadGRDECL_ReadGRDECL_H_

#include <iostream>
#include <fstream>

#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QTextEdit>
#include <QLineEdit>
#include <QFile>
#include <QString>
#include <QStringList>
#include <QLabel>
#include <QFileDialog>
#include <QTextStream>
#include <QVector>
#include <QVector3D>
#include <QQuickView>


class ReadGRDECL : public QWidget
{
	Q_OBJECT

public:
    ReadGRDECL(QWidget *parent=nullptr);
    virtual ~ReadGRDECL();

private:

	QTextEdit *qte_input_mesh;
	QLineEdit *qle_mesh_object_name;

	QTextEdit *qte_converted_mesh;

	// Path to workspace: TODO to modify before merging
	std::string path = "/data/PLI/NKDEEP/sytuan/Libs/ECL2VTK/";
	//std::string inputFileName; 

	void load_file_to_qtextedit(QString const, QTextEdit*);
	void save_qtextedit_to_file(QString const, QTextEdit*);

private slots:

	void convert_mesh();
	void load_input_mesh();
	void save_converted_mesh();
	void view_converted_mesh();
};


#endif /* NEXTVISION_SRC_WIDGET_WReadGRDECL_ReadGRDECL_H_ */


