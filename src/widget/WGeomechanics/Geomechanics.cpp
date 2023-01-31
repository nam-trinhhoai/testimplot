/*
 *
 *
 *  Created on: 24 May 2022
 *      Author: l0359127
 */

#include "Geomechanics.h"

Geomechanics::Geomechanics(QWidget *parent)
{
	this->setWindowTitle("Geomechanics");
	this->setMinimumSize(1012, 683);

	// Mesh script widgets
	QVBoxLayout *qbl_input_mesh = new QVBoxLayout();

	// Mesh script controls
	QHBoxLayout *qbl_input_mesh_control = new QHBoxLayout();
	QLabel *lbl_input_mesh = new QLabel("Input Mesh");	
	
	QPushButton *qpb_load_input_mesh = new QPushButton("Load GRDECL");
	connect(qpb_load_input_mesh, SIGNAL(clicked()), this, SLOT(load_input_mesh()));

	qbl_input_mesh_control->addWidget(lbl_input_mesh);
	qbl_input_mesh_control->addWidget(qpb_load_input_mesh);	
			
	// Mesh script editor
	qte_input_mesh = new QTextEdit();


	qbl_input_mesh->addLayout(qbl_input_mesh_control);
	qbl_input_mesh->addWidget(qte_input_mesh);

	// converted mesh widgets
	

	QLabel *lbl_converted_mesh = new QLabel("Output Mesh");	

	QPushButton *qpb_save_converted_mesh = new QPushButton("Save mesh");
	connect(qpb_save_converted_mesh, SIGNAL(clicked()), this, SLOT(save_converted_mesh()));

	QPushButton *qpb_view_converted_mesh = new QPushButton("Preview");
	connect(qpb_view_converted_mesh, SIGNAL(clicked()), this, SLOT(view_converted_mesh()));

	QHBoxLayout *qbl_converted_mesh_control = new QHBoxLayout();
	qbl_converted_mesh_control->addWidget(lbl_converted_mesh);
	qbl_converted_mesh_control->addWidget(qpb_save_converted_mesh);
	qbl_converted_mesh_control->addWidget(qpb_view_converted_mesh);

	qte_converted_mesh = new QTextEdit();


	QVBoxLayout *qbl_converted_mesh = new QVBoxLayout();
	qbl_converted_mesh->addLayout(qbl_converted_mesh_control);
	qbl_converted_mesh->addWidget(qte_converted_mesh);





	// Text splitter
	QHBoxLayout *textLayout = new QHBoxLayout;

	textLayout->addLayout(qbl_input_mesh);
	textLayout->addLayout(qbl_converted_mesh);
	
	// Main layout
	QVBoxLayout *mainLayout = new QVBoxLayout(this);

	QPushButton *qpb_convert_mesh = new QPushButton("Convert Mesh");
	connect(qpb_convert_mesh, SIGNAL(clicked()), this, SLOT(convert_mesh()));
		
	mainLayout->addLayout(textLayout);
	mainLayout->addWidget(qpb_convert_mesh);	
}

Geomechanics::~Geomechanics()
{
}

void Geomechanics::convert_mesh()
{	

	// Create tmp input file for conversion
	std::ofstream ofile (path + "tmp.GRDECL");
  
	ofile << qte_input_mesh->toPlainText().toStdString() << std::endl;

	ofile.close();

	// Launch Lagrit
	system( ("cd " + path + " && python readGRDECL.py tmp.GRDECL").c_str() );

	// Show output mesh
	load_file_to_qtextedit((path + "tmp.inp").c_str(), qte_converted_mesh);

	// Remove tmp files
	system( ("rm " + path + "tmp.*").c_str() );	
}

void Geomechanics::load_input_mesh()
{
	QString fileName = QFileDialog::getOpenFileName(this,
    	tr("Open GRDECL mesh"), path.c_str(), tr("Eclipse Files (*.GRDECL)"));
	
	load_file_to_qtextedit(fileName, qte_input_mesh);

	// Update input file name
	//inputFileName = fileName.toStdString();
}

void Geomechanics::load_file_to_qtextedit(QString const fileName, QTextEdit *qte)
{
	QFile file(fileName);
	file.open(QFile::ReadOnly | QFile::Text);

	qte->setText(file.readAll());
	file.close();
}

void Geomechanics::save_converted_mesh()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save mesh file"), path.c_str(), tr("Mesh file (*.inp)"));

	save_qtextedit_to_file(fileName, qte_converted_mesh);
}

void Geomechanics::save_qtextedit_to_file(QString const fileName, QTextEdit *qte)
{
	QFile file(fileName);
	file.open(QFile::WriteOnly | QFile::Text);

	QTextStream stream(&file);
	stream << qte->toPlainText();

	file.flush();
	file.close();
}

void Geomechanics::view_converted_mesh()
{
	system("/data/appli_PITSI/MAJIX2018/PROD/ParaviewLauncherGUI/paraviewLauncherGUI.sh");
}




