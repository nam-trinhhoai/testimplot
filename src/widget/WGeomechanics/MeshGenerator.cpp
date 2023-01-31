/*
 *
 *
 *  Created on: 10 May 2022
 *      Author: l0359127
 */

#include "MeshGenerator.h"

MeshGenerator::MeshGenerator(QWidget *parent)
{
	this->setWindowTitle("Mesh Generator");
	this->setMinimumSize(1012, 683);

	// Mesh script widgets
	QVBoxLayout *qbl_mesh_script = new QVBoxLayout();

	// Mesh script controls
	QHBoxLayout *qbl_mesh_script_control = new QHBoxLayout();
	QLabel *lbl_mesh_script = new QLabel("Mesh Script");	
	
	QPushButton *qpb_load_mesh_script = new QPushButton("Load script");
	connect(qpb_load_mesh_script, SIGNAL(clicked()), this, SLOT(trt_load_mesh_script()));

	QPushButton *qpb_save_mesh_script = new QPushButton("Save script");
	connect(qpb_save_mesh_script, SIGNAL(clicked()), this, SLOT(trt_save_mesh_script()));
	
	qbl_mesh_script_control->addWidget(lbl_mesh_script);
	qbl_mesh_script_control->addWidget(qpb_load_mesh_script);	
	qbl_mesh_script_control->addWidget(qpb_save_mesh_script);
			
	// Mesh script editor
	qte_mesh_script = new QTextEdit();

	// Mesh object name
	QHBoxLayout *qbl_mesh_object_name = new QHBoxLayout();

	QLabel *lbl_mesh_object_name = new QLabel("Mesh object name");
	qle_mesh_object_name = new QLineEdit();

	qbl_mesh_object_name->addWidget(lbl_mesh_object_name);
	qbl_mesh_object_name->addWidget(qle_mesh_object_name);	

	qbl_mesh_script->addLayout(qbl_mesh_script_control);
	qbl_mesh_script->addWidget(qte_mesh_script);
	qbl_mesh_script->addLayout(qbl_mesh_object_name);




	// Generated mesh widgets
	

	QLabel *lbl_generated_mesh = new QLabel("Generated Mesh");	

	QPushButton *qpb_save_generated_mesh = new QPushButton("Save mesh");
	connect(qpb_save_generated_mesh, SIGNAL(clicked()), this, SLOT(trt_save_generated_mesh()));

	QPushButton *qpb_view_generated_mesh = new QPushButton("Preview");
	connect(qpb_view_generated_mesh, SIGNAL(clicked()), this, SLOT(trt_view_generated_mesh()));

	QHBoxLayout *qbl_generated_mesh_control = new QHBoxLayout();
	qbl_generated_mesh_control->addWidget(lbl_generated_mesh);
	qbl_generated_mesh_control->addWidget(qpb_save_generated_mesh);
	qbl_generated_mesh_control->addWidget(qpb_view_generated_mesh);

	qte_generated_mesh = new QTextEdit();


	QVBoxLayout *qbl_generated_mesh = new QVBoxLayout();
	qbl_generated_mesh->addLayout(qbl_generated_mesh_control);
	qbl_generated_mesh->addWidget(qte_generated_mesh);





	// Text splitter
	QHBoxLayout *textLayout = new QHBoxLayout;

	textLayout->addLayout(qbl_mesh_script);
	textLayout->addLayout(qbl_generated_mesh);
	
	// Main layout
	QVBoxLayout *mainLayout = new QVBoxLayout(this);

	QPushButton *qpb_generate_mesh = new QPushButton("Generate Mesh");
	connect(qpb_generate_mesh, SIGNAL(clicked()), this, SLOT(trt_generate_mesh()));
		
	mainLayout->addLayout(textLayout);
	mainLayout->addWidget(qpb_generate_mesh);	
}

MeshGenerator::~MeshGenerator()
{
}

void MeshGenerator::trt_generate_mesh()
{
	// Create input file for Lagrit
	std::ofstream ofile (lagrit_path + "tmp/tmp.lgi");
  
	ofile << qte_mesh_script->toPlainText().toStdString() << std::endl;

	ofile << "dump / tmp.inp / " << qle_mesh_object_name->text().toStdString() << std::endl;
	
	ofile.close();

	// Launch Lagrit
	system( ("cd " + lagrit_path + "tmp/ && " + lagrit_path + "LaGriT/build/lagrit < tmp.lgi").c_str() );

	// Show output mesh
	load_file_to_qtextedit((lagrit_path + "tmp/tmp.inp").c_str(), qte_generated_mesh);

	// Remove tmp files
	system( ("rm " + lagrit_path + "tmp/*").c_str() );	
}

void MeshGenerator::trt_load_mesh_script()
{
	QString fileName = QFileDialog::getOpenFileName(this,
    	tr("Open Lagrit script"), lagrit_path.c_str(), tr("Lagrit Files (*.lgi *.txt)"));
	
	load_file_to_qtextedit(fileName, qte_mesh_script);
}

void MeshGenerator::load_file_to_qtextedit(QString const fileName, QTextEdit *qte)
{
	QFile file(fileName);
	file.open(QFile::ReadOnly | QFile::Text);

	qte->setText(file.readAll());
	file.close();
}

void MeshGenerator::trt_save_mesh_script()
{
	QString fileName = QFileDialog::getSaveFileName(this, 
		tr("Save Lagrit script"), lagrit_path.c_str(), tr("Lagrit Files (*.lgi)"));

	save_qtextedit_to_file(fileName, qte_mesh_script);
}

void MeshGenerator::trt_save_generated_mesh()
{
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save mesh file"), lagrit_path.c_str(), tr("Mesh file (*.inp)"));

	save_qtextedit_to_file(fileName, qte_generated_mesh);
}

void MeshGenerator::save_qtextedit_to_file(QString const fileName, QTextEdit *qte)
{
	QFile file(fileName);
	file.open(QFile::WriteOnly | QFile::Text);

	QTextStream stream(&file);
	stream << qte->toPlainText();

	file.flush();
	file.close();
}

void MeshGenerator::trt_view_generated_mesh()
{
/**
	QString const fileName = QFileDialog::getOpenFileName(this,
		tr("Load mesh file"), lagrit_path.c_str(), tr("Mesh file (*.inp)"));


load_file_to_qtextedit(fileName, qte_generated_mesh);


	QFile file(fileName);
	file.open(QFile::ReadOnly);

	QTextStream stream(&file);

	QStringList const firstLineList = stream.readLine().split(" ", QString::SkipEmptyParts);

	int const nNodes = firstLineList[0].toInt();
	int const nElements = firstLineList[1].toInt();

	QVector<QVector3D> nodes;
	
	// Get nodes coordinates
	nodes.resize(nNodes);

	for(int i=0; i<nNodes; i++)
	{
		QStringList const lineList = stream.readLine().split(" ", QString::SkipEmptyParts);
		
		nodes[i] = QVector3D(lineList[1].toDouble(),
							 lineList[2].toDouble(),
							 lineList[3].toDouble());
	}

	// Get faces triangles node indexes
	
	int nTriangles = nElements*6*2*3; // Each hex element has 6 faces, 
									  // each face is devided into two triangles, 
									  // each triangle has three nodes
	
	QVector<int> nodeIndexes_for_faces_triangles;
	nodeIndexes_for_faces_triangles.resize(nTriangles);
	int idx = 0;

	for(int i=0; i<nElements; i++)
	{
		QStringList lineList = stream.readLine().split(" ", QString::SkipEmptyParts);

		// Get the node indexes of each Hex element
		QVector<int> nodeIndexes(8);
		for(int j=0; j<8; j++)
		{
			nodeIndexes[j] = lineList[j+3].toInt();
		}
		
		// Add element nodes to the liste of face nodes
		// Top face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[1];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[3];

		// Bottom face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[4];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[4];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[5];

		// Left face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[3];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[4];

		// Right face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[1];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[5];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[1];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];

		// Front face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[4];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[5];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[5];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[1];

		// Back face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[3];
	}


	file.close();
*/

	

	QQuickView *view = new QQuickView();
	
	view->setSource(QUrl(QStringLiteral("/data/PLI/NKDEEP/sytuan/Libs/Lagrit/View3dResource.qml")));

	//ViewMesh *viewer3D = new ViewMesh(this);

    //viewer3D->sceneModifier()->addTriangleMeshCustomMaterial(name, m_meshVector);
    //viewer3D->show();
}



