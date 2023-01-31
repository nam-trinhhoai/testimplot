/*
 *
 *
 *  Created on: 10 May 2022
 *      Author: l0359127
 */


#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_MESHGENERATOR_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_MESHGENERATOR_H_

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

#include "ViewMesh.h"


class MeshGenerator : public QWidget
{
	Q_OBJECT

public:
    MeshGenerator(QWidget *parent=nullptr);
    virtual ~MeshGenerator();

private:

	QTextEdit *qte_mesh_script;
	QLineEdit *qle_mesh_object_name;

	QTextEdit *qte_generated_mesh;

	// Path to Lagrit workspace: TODO to modify before merging
	std::string const lagrit_path = "/data/PLI/NKDEEP/sytuan/Libs/Lagrit/"; 

	void load_file_to_qtextedit(QString const, QTextEdit*);
	void save_qtextedit_to_file(QString const, QTextEdit*);

private slots:

	void trt_generate_mesh();
	void trt_load_mesh_script();
	void trt_save_mesh_script();
	void trt_save_generated_mesh();
	void trt_view_generated_mesh();
};


#endif /* NEXTVISION_SRC_WIDGET_WGEOMECHANICS_MESHGENERATOR_H_ */


