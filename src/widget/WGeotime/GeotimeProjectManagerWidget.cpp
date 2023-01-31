/*
 *
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */


#include <QTableView>
#include <QHeaderView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QRadioButton>
#include <QLabel>
#include <QPainter>
#include <QChart>
#include <QLineEdit>
#include <QToolButton>
#include <QLineSeries>
#include <QScatterSeries>
#include <QtCharts>
#include <QRandomGenerator>
#include <QComboBox>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <QFile>

#include <QVBoxLayout>
#include <QProcess>
#include <QDebug>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include <sys/stat.h>
#include "Xt.h"
#include "GeotimeProjectManagerWidget.h"
#include <freeHorizonManager.h>
#include "SeismicManager.h"
#include "globalconfig.h"
#include <seismicDatabaseManager.h>
#include <wellsDatabaseManager.h>
#include <geotimepath.h>
#include "wellbore.h"



namespace fs = boost::filesystem;

#define DEBUG00 fprintf(stderr, "%s %d\n", __FILE__, __LINE__);

// #define __LINUX__

using namespace std;


// namespace XCom = process::XCom;

GeotimeProjectManagerWidget::GeotimeProjectManagerWidget(QWidget* parent) :
		QWidget(parent) {


	QPushButton *qpb_debug = new QPushButton("DEBUG");


    int qtmargin = 10;
    qgb_projectmanager = new QGroupBox(this);
	qgb_projectmanager->setTitle("... - ...");
	// qgb_projectmanager->setGeometry(QRect(qtmargin, 50, 370+150, 500+300));

    QHBoxLayout* qhb_labelprojectname = new QHBoxLayout;
    label_projectname = new QLabel("... - ...");
    qhb_labelprojectname->addWidget(label_projectname);

    QHBoxLayout* qhb_projecttype = new QHBoxLayout;
    label_projecttype = new QLabel();
    label_projecttype->setText("Project type:");
    cb_projecttype = new QComboBox();
    this->cb_projecttype->addItem("None");

    GlobalConfig& config = GlobalConfig::getConfig();
    const std::vector<std::pair<QString, QString>>& dir_projects = config.dirProjects();
    for (const std::pair<QString, QString>& pair : dir_projects) {
    	this->cb_projecttype->addItem(pair.first);
    }
    this->cb_projecttype->addItem("USER");

    QHBoxLayout* qhb_projectcustompath = new QHBoxLayout;
    lineedit_custompath = new QLineEdit("");
    lineedit_custompath->setEnabled(false);
    label_custom_path = new QLabel("path:");
    label_custom_path->setEnabled(false);
    qpb_custompath = new QPushButton("ok");
    qpb_custompath->setEnabled(false);

    qhb_projectcustompath->addWidget(label_custom_path);
    qhb_projectcustompath->addWidget(lineedit_custompath);
    qhb_projectcustompath->addWidget(qpb_custompath);






    this->cb_projecttype->setCurrentIndex(0); 
    this->cb_projecttype->setStyleSheet("QComboBox::item{height: 20px}");

    // qhb_projecttype->addWidget(label_projecttype);
    qhb_projecttype->addWidget(cb_projecttype);

    QHBoxLayout* qhb_type1 = new QHBoxLayout;
    chkbx_culturals = new QCheckBox("Culturals");
    chkbx_wells = new QCheckBox("Wells / wellbores");
    chkbx_neurons = new QCheckBox("Neurons");
    chkbx_horizons = new QCheckBox("Horizons");
    qhb_type1->addWidget(chkbx_culturals);
    qhb_type1->addWidget(chkbx_wells);
    qhb_type1->addWidget(chkbx_neurons);
    qhb_type1->addWidget(chkbx_horizons);

    this->lw_projetlist = new QListWidget();
    lineedit_projectsearch = new QLineEdit();
    QGroupBox *qgb_projectlist = new QGroupBox();
    qgb_projectlist->setTitle("Project");
    QVBoxLayout* mainLayout_projectlist = new QVBoxLayout(qgb_projectlist);
    mainLayout_projectlist->addLayout(qhb_projecttype);
    mainLayout_projectlist->addLayout(qhb_projectcustompath);
    mainLayout_projectlist->addWidget(lineedit_projectsearch);
    mainLayout_projectlist->addWidget(lw_projetlist);    

    this->lw_surveylist = new QListWidget();
    lineedit_surveysearch = new QLineEdit();
    QGroupBox *qgb_survey = new QGroupBox();
    qgb_survey->setTitle("Survey");
    QVBoxLayout* mainLayout_surveylist = new QVBoxLayout(qgb_survey);
    mainLayout_surveylist->addWidget(lineedit_surveysearch);
    mainLayout_surveylist->addWidget(lw_surveylist);    


/*
    lw_cultural = new QListWidget();
    lw_cultural->setSelectionMode(QAbstractItemView::MultiSelection);
    lineedit_culturalsearch = new QLineEdit();
    QGroupBox *qgb_culturals = new QGroupBox();
    QVBoxLayout* mainLayout_culturals = new QVBoxLayout(qgb_culturals);
    mainLayout_culturals->addWidget(lineedit_culturalsearch);
    mainLayout_culturals->addWidget(lw_cultural);
    */

    QVBoxLayout *buttons_culturals_basket_vbox = new QVBoxLayout();
    QPushButton *culturals_basket_add = new QPushButton(">>"), *culturals_basket_sub = new QPushButton("<<");
    buttons_culturals_basket_vbox->addWidget(culturals_basket_add);
    buttons_culturals_basket_vbox->addWidget(culturals_basket_sub);
    lw_cultural = new QListWidget();
    lw_cultural_basket = new QListWidget();
    lw_cultural->setSelectionMode(QAbstractItemView::MultiSelection);
    lw_cultural_basket->setSelectionMode(QAbstractItemView::MultiSelection);
    lineedit_culturalsearch = new QLineEdit();
    QGroupBox *qgb_culturals = new QGroupBox();
    QHBoxLayout *lw_cultural_hbox = new QHBoxLayout();
    lw_cultural_hbox->addWidget(lw_cultural);
    lw_cultural_hbox->addLayout(buttons_culturals_basket_vbox);
    lw_cultural_hbox->addWidget(lw_cultural_basket);
    QVBoxLayout* mainLayout_culturals = new QVBoxLayout(qgb_culturals);
    mainLayout_culturals->addWidget(lineedit_culturalsearch);
    // mainLayout_seismic->addWidget(lw_seismic);
    mainLayout_culturals->addLayout(lw_cultural_hbox);
    QPushButton *qpb_cultural_database_update = new QPushButton("Culturals DataBase update");
    mainLayout_culturals->addWidget(qpb_cultural_database_update);


    lw_wells = new QListWidget();
    lw_wellsbasket = new QListWidget();
    QVBoxLayout *qvb_wellbuttons = new QVBoxLayout;
    QPushButton *qpb_well_basket_add = new QPushButton(">>");
    QPushButton *qpb_well_basket_sub = new QPushButton("<<");
    qvb_wellbuttons->addWidget(qpb_well_basket_add);
    qvb_wellbuttons->addWidget(qpb_well_basket_sub);
    // lw_wells->setSelectionMode(QAbstractItemView::MultiSelection);
    lineedit_wellssearch = new QLineEdit();
    lineedit_wellssearch->setToolTip("command line example log=sometext;tf2p=sometext;picks=sometext");
    lw_wellbore = new QListWidget();

    lw_wells->setSelectionMode(QAbstractItemView::MultiSelection);
    lw_wellsbasket->setSelectionMode(QAbstractItemView::ExtendedSelection);
    linedit_wellboresearch = new QLineEdit();
    QGroupBox *qgb_wells = new QGroupBox();
    QVBoxLayout* mainLayout_wells = new QVBoxLayout(qgb_wells);    
    mainLayout_wells->addWidget(lineedit_wellssearch);

    QHBoxLayout *qhb_well = new QHBoxLayout;
    qhb_well->addWidget(lw_wells);
    qhb_well->addLayout(qvb_wellbuttons);
    qhb_well->addWidget(lw_wellsbasket);
    mainLayout_wells->addLayout(qhb_well);


    // mainLayout_wells->addWidget(lw_wells);

    // mainLayout_wells->addWidget(linedit_wellboresearch);
    // mainLayout_wells->addWidget(lw_wellbore);
    
    QGroupBox *qgb_wellog = new QGroupBox();
    QGroupBox *qgb_tfp2 = new QGroupBox();
    QGroupBox *qgb_picks = new QGroupBox();

    QVBoxLayout* qvb_welllog = new QVBoxLayout(qgb_wellog);
    linedit_welllogsearch = new QLineEdit();

    // QHBoxLayout* qhb_welllog = new QHBoxLayout(qgb_wellog);
    // QVBoxLayout *qvb_welllog_button = new QVBoxLayout(qgb_wellog);
    QHBoxLayout* qhb_welllog = new QHBoxLayout;
    QVBoxLayout *qvb_welllog_button = new QVBoxLayout;

    QPushButton *qpb_welllog_basket_add = new QPushButton(">>");
    QPushButton *qpb_welllog_basket_sub = new QPushButton("<<");
    qvb_welllog_button->addWidget(qpb_welllog_basket_add);
    qvb_welllog_button->addWidget(qpb_welllog_basket_sub);

    qlw_welllog = new QListWidget();
    qlw_welllog->setSelectionMode(QAbstractItemView::MultiSelection);

    qlw_welllog_basket = new QListWidget();
    qlw_welllog_basket->setSelectionMode(QAbstractItemView::MultiSelection);
    // qtw_welllog_basket = new QTableWidget(10, 2);
    // qtw_welllog_basket->setHorizontalHeaderItem(0, new QTableWidgetItem("Well"));
    // qtw_welllog_basket->setHorizontalHeaderItem(1, new QTableWidgetItem("Log"));

    qhb_welllog->addWidget(qlw_welllog);
    qhb_welllog->addLayout(qvb_welllog_button);
    qhb_welllog->addWidget(qlw_welllog_basket);
    // qhb_welllog->addWidget(qtw_welllog_basket);

    qvb_welllog->addWidget(linedit_welllogsearch);
    qvb_welllog->addLayout(qhb_welllog);

    QVBoxLayout* qvb_welltf2p = new QVBoxLayout(qgb_tfp2);
    linedit_welltf2psearch = new QLineEdit();
    QHBoxLayout* qhb_welltf2p = new QHBoxLayout;
    QVBoxLayout *qvb_welltf2p_button = new QVBoxLayout;
    QPushButton *qpb_welltf2p_basket_add = new QPushButton(">>");
    QPushButton *qpb_welltf2p_basket_sub = new QPushButton("<<");
    qvb_welltf2p_button->addWidget(qpb_welltf2p_basket_add);
    qvb_welltf2p_button->addWidget(qpb_welltf2p_basket_sub);

    qlw_welltf2p = new QListWidget();
    qlw_welltf2p->setSelectionMode(QAbstractItemView::MultiSelection);
    qlw_welltf2p_basket = new QListWidget();
    qlw_welltf2p_basket->setSelectionMode(QAbstractItemView::MultiSelection);
    qhb_welltf2p->addWidget(qlw_welltf2p);
    qhb_welltf2p->addLayout(qvb_welltf2p_button);
    qhb_welltf2p->addWidget(qlw_welltf2p_basket);

    qvb_welltf2p->addWidget(linedit_welltf2psearch);
    qvb_welltf2p->addLayout(qhb_welltf2p);

    QVBoxLayout* qvb_wellpicks = new QVBoxLayout(qgb_picks);
    linedit_wellpickssearch = new QLineEdit();
    QHBoxLayout* qhb_wellpicks = new QHBoxLayout;
    QVBoxLayout *qvb_wellpicks_button = new QVBoxLayout;
    QPushButton *qpb_wellpicks_basket_add = new QPushButton(">>");
    QPushButton *qpb_wellpicks_basket_sub = new QPushButton("<<");
    qvb_wellpicks_button->addWidget(qpb_wellpicks_basket_add);
    qvb_wellpicks_button->addWidget(qpb_wellpicks_basket_sub);

    qlw_wellpicks = new QListWidget();
    qlw_wellpicks->setSelectionMode(QAbstractItemView::MultiSelection);
    qlw_wellpicks_basket = new QListWidget();
    qlw_wellpicks_basket->setSelectionMode(QAbstractItemView::MultiSelection);
    qhb_wellpicks->addWidget(qlw_wellpicks);
    qhb_wellpicks->addLayout(qvb_wellpicks_button);
    qhb_wellpicks->addWidget(qlw_wellpicks_basket);

    qvb_wellpicks->addWidget(linedit_wellpickssearch);
    qvb_wellpicks->addLayout(qhb_wellpicks);

    QTabWidget *tabw_wells = new QTabWidget(); int idx1 = 1;
    tabw_wells->insertTab(idx1++, qgb_wellog, QString("Log"));
    tabw_wells->insertTab(idx1++, qgb_tfp2, "TF2P");
    // tabw_wells->insertTab(idx1++, qgb_picks, "Picks");
    mainLayout_wells->addWidget(tabw_wells);
    QPushButton *qpb_well_database_update = new QPushButton("Wells DataBase update");
    mainLayout_wells->addWidget(qpb_well_database_update);


    lw_neurons = new QListWidget(); 
    lw_neurons->setSelectionMode(QAbstractItemView::MultiSelection);
    lineedit_neuronssearch = new QLineEdit();
    QGroupBox *qgb_neurons = new QGroupBox();
    QVBoxLayout* mainLayout_neurons = new QVBoxLayout(qgb_neurons);
    mainLayout_neurons->addWidget(lineedit_neuronssearch);
    mainLayout_neurons->addWidget(lw_neurons); 

    QVBoxLayout *buttons_horizons_basket_vbox = new QVBoxLayout();
    QPushButton *horizons_basket_add = new QPushButton(">>"), *horizons_basket_sub = new QPushButton("<<");
    buttons_horizons_basket_vbox->addWidget(horizons_basket_add);
    buttons_horizons_basket_vbox->addWidget(horizons_basket_sub);

    lw_horizons = new QListWidget(); 
    lw_horizons_basket = new QListWidget();
    lw_horizons->setSelectionMode(QAbstractItemView::MultiSelection); 
    lw_horizons_basket->setSelectionMode(QAbstractItemView::MultiSelection); 
    lineedit_horizonssearch = new QLineEdit();
    QGroupBox *qgb_horizons = new QGroupBox();    
    QHBoxLayout *lw_horizons_hbox = new QHBoxLayout();
    lw_horizons_hbox->addWidget(lw_horizons);
    lw_horizons_hbox->addLayout(buttons_horizons_basket_vbox);
    lw_horizons_hbox->addWidget(lw_horizons_basket);
    QVBoxLayout* mainLayout_horizons = new QVBoxLayout(qgb_horizons);
    mainLayout_horizons->addWidget(lineedit_horizonssearch);
    mainLayout_horizons->addLayout(lw_horizons_hbox);
    QPushButton *qpb_horizon_database_update = new QPushButton("Horizons DataBase update");
    mainLayout_horizons->addWidget(qpb_horizon_database_update);


    QVBoxLayout *buttons_seismic_basket_vbox = new QVBoxLayout();
    QPushButton *seismic_basket_add = new QPushButton(">>"), *seismic_basket_sub = new QPushButton("<<");
    buttons_seismic_basket_vbox->addWidget(seismic_basket_add);
    buttons_seismic_basket_vbox->addWidget(seismic_basket_sub);

    lw_seismic = new QListWidget();  
    lw_seismic_basket = new QListWidget();
    lw_seismic->setSelectionMode(QAbstractItemView::MultiSelection);
    lw_seismic_basket->setSelectionMode(QAbstractItemView::MultiSelection);
    lineedit_seismicsearch = new QLineEdit();
    QGroupBox *qgb_seismic = new QGroupBox();
    QHBoxLayout *lw_seismic_hbox = new QHBoxLayout();
    lw_seismic_hbox->addWidget(lw_seismic);
    lw_seismic_hbox->addLayout(buttons_seismic_basket_vbox);
    lw_seismic_hbox->addWidget(lw_seismic_basket);
    QVBoxLayout* mainLayout_seismic = new QVBoxLayout(qgb_seismic);
    mainLayout_seismic->addWidget(lineedit_seismicsearch);
    // mainLayout_seismic->addWidget(lw_seismic);
    mainLayout_seismic->addLayout(lw_seismic_hbox);
    QPushButton *qpb_seismic_database_update = new QPushButton("Seismic DataBase update");
    mainLayout_seismic->addWidget(qpb_seismic_database_update);

    QGroupBox *qgb_cubeRgt2RGB = new QGroupBox;
    lineedit_rgt2rgtsearch= new QLineEdit;

    QVBoxLayout* mainLayout_rgt2rgb = new QVBoxLayout(qgb_cubeRgt2RGB);
    QHBoxLayout *mainLayout_rgt2rgb_1 = new QHBoxLayout;
    qlw_rgb = new QListWidget;
    qlw_rgb_basket = new QListWidget;
    qlw_rgb->setSelectionMode(QAbstractItemView::MultiSelection);
    qlw_rgb_basket->setSelectionMode(QAbstractItemView::MultiSelection);

    QVBoxLayout* mainLayout_rgt2rgb_2 = new QVBoxLayout;
    QPushButton *qpb_rgb_basket_add = new QPushButton(">>");
    QPushButton *qpb_rgb_basket_sub = new QPushButton("<<");
    mainLayout_rgt2rgb_2->addWidget(qpb_rgb_basket_add);
    mainLayout_rgt2rgb_2->addWidget(qpb_rgb_basket_sub);
    mainLayout_rgt2rgb_1->addWidget(qlw_rgb);
    mainLayout_rgt2rgb_1->addLayout(mainLayout_rgt2rgb_2);
    mainLayout_rgt2rgb_1->addWidget(qlw_rgb_basket);
    mainLayout_rgt2rgb->addWidget(lineedit_rgt2rgtsearch);
    mainLayout_rgt2rgb->addLayout(mainLayout_rgt2rgb_1);

//sylvain
    QVBoxLayout *buttons_nurbs_basket_vbox = new QVBoxLayout();
      QPushButton *nurbs_basket_add = new QPushButton(">>"), *nurbs_basket_sub = new QPushButton("<<");
      buttons_nurbs_basket_vbox->addWidget(nurbs_basket_add);
      buttons_nurbs_basket_vbox->addWidget(nurbs_basket_sub);

      lw_nurbs = new QListWidget();
      lw_nurbs_basket = new QListWidget();
      lw_nurbs->setSelectionMode(QAbstractItemView::MultiSelection);
      lw_nurbs_basket->setSelectionMode(QAbstractItemView::MultiSelection);
      lineedit_nurbssearch = new QLineEdit();
      QGroupBox *qgb_nurbs = new QGroupBox();
      QHBoxLayout *lw_nurbs_hbox = new QHBoxLayout();
      lw_nurbs_hbox->addWidget(lw_nurbs);
      lw_nurbs_hbox->addLayout(buttons_nurbs_basket_vbox);
      lw_nurbs_hbox->addWidget(lw_nurbs_basket);
      QVBoxLayout* mainLayout_nurbs = new QVBoxLayout(qgb_nurbs);
      mainLayout_nurbs->addWidget(lineedit_nurbssearch);
      // mainLayout_seismic->addWidget(lw_seismic);
      mainLayout_nurbs->addLayout(lw_nurbs_hbox);
      QPushButton *qpb_nurbs_database_update = new QPushButton("Nurbs DataBase update");
      mainLayout_nurbs->addWidget(qpb_nurbs_database_update);

    //Sylvain

	QGroupBox *qgb_picks2 = new QGroupBox();
	QVBoxLayout *picksLayout = new QVBoxLayout(qgb_picks2);
	m_picksManager = new PicksManager();
	// m_projectManager->setPicksManager(m_picksManager);
	picksLayout->addWidget(m_picksManager);

    this->tabw_table1 = new QTabWidget(); int idx0 = 1;
    // tabw_table1->insertTab(idx0++, qgb_projectlist, QIcon(QString("")), "Project");
    // tabw_table1->insertTab(idx0++, qgb_survey, QIcon(QString("")), "Survey");
    tabw_table1->insertTab(idx0++, qgb_seismic, QIcon(QString("")), "Seismic");
    tabw_table1->insertTab(idx0++, qgb_horizons, QIcon(QString("")), "Horizons");     
    tabw_table1->insertTab(idx0++, qgb_culturals, QIcon(QString("")), "Culturals");
    tabw_table1->insertTab(idx0++, qgb_wells, QIcon(QString("")), "Wells / wellbores");
    tabw_table1->insertTab(idx0++, qgb_picks2, QIcon(QString("")), "Picks");
    tabw_table1->insertTab(idx0++, qgb_neurons, QIcon(QString("")), "Neurons");
    tabw_table1->insertTab(idx0++, qgb_cubeRgt2RGB, QIcon(QString("")), "RGT --> RGB");
    tabw_table1->insertTab(idx0++, qgb_nurbs, QIcon(QString("")), "Nurbs");

    QHBoxLayout* qhb_projectsurvey = new QHBoxLayout();
    qhb_projectsurvey->addWidget(qgb_projectlist);
    qhb_projectsurvey->addWidget(qgb_survey); 
        
    QVBoxLayout* mainLayout = new QVBoxLayout(this);//(qgb_projectmanager);
    // mainLayout->addWidget(qpb_debug);
    mainLayout->addWidget(qgb_projectmanager);
    mainLayout->addLayout(qhb_projectsurvey);
    // mainLayout->addWidget(tabw_table1);

// <<<<<<< HEAD
// <<<<<<< HEAD
// =======
// >>>>>>> ec7765fe1da31ea75cd73de9c197af6e162e1ac5
    /*
    for (int i=0; i<getDirProjects().size(); i++)
    {
        QString path = get_project_path(i+1);
        this->project_list[i+1] = get_dirlist2(path);
        fprintf(stderr, "reading path %d / 3 [ %s ]\n", i, path.toStdString().c_str());
    }
    */
    project_list.resize(config.dirProjects().size()+2);

    tab_seismic = true;
    tab_horizons = true;
    tab_wells = true;
    tab_culturals = true;
    tab_neurons = true;
    tab_picks = true;
    tab_nurbs = true;

    // project_list_init();

    connect(cb_projecttype, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_projecttypeclick(int)));
    connect(lw_projetlist, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_projetlistClick(QListWidgetItem*)));
    connect(lineedit_projectsearch, SIGNAL(textChanged(QString)), this, SLOT(trt_projectsearchchange(QString)));  
    connect(lineedit_surveysearch, SIGNAL(textChanged(QString)), this, SLOT(trt_surveysearchchange(QString)));  
    connect(lw_surveylist, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_surveylistClick(QListWidgetItem*)));
    connect(lineedit_wellssearch, SIGNAL(textChanged(QString)), this, SLOT(trt_wellssearchchange(QString)));  
    connect(lineedit_horizonssearch, SIGNAL(textChanged(QString)), this, SLOT(trt_horizonssearchchange(QString)));      
    connect(lineedit_seismicsearch, SIGNAL(textChanged(QString)), this, SLOT(trt_seismicsearchchange(QString)));
    connect(lineedit_culturalsearch, SIGNAL(textChanged(QString)), this, SLOT(trt_culturalssearchchange(QString)));
    connect(lineedit_neuronssearch, SIGNAL(textChanged(QString)), this, SLOT(trt_neuronssearchchange(QString)));

    connect(lineedit_rgt2rgtsearch, SIGNAL(textChanged(QString)), this, SLOT(trt_rgbsearchchange(QString)));


    connect(lw_wells, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_welllistclick(QListWidgetItem*)));

    //connect(lw_wellsbasket, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_wellbasketlistclick(QListWidgetItem*)));
    connect(lw_wellsbasket, SIGNAL(itemSelectionChanged()), this, SLOT(trt_wellbasketlistselectionchanged()));

    connect(linedit_welllogsearch, SIGNAL(textChanged(QString)), this, SLOT(trt_welllogsearchchange(QString)));
    connect(linedit_welltf2psearch, SIGNAL(textChanged(QString)), this, SLOT(trt_welltf2psearchchange(QString)));
    connect(linedit_wellpickssearch, SIGNAL(textChanged(QString)), this, SLOT(trt_wellpickssearchchchange(QString)));

    connect(chkbx_culturals, SIGNAL(stateChanged(int)), this, SLOT(trt_chkbx1(int)));
    connect(chkbx_wells, SIGNAL(stateChanged(int)), this, SLOT(trt_chkbx1(int)));
    connect(chkbx_neurons, SIGNAL(stateChanged(int)), this, SLOT(trt_chkbx1(int)));
    connect(chkbx_horizons, SIGNAL(stateChanged(int)), this, SLOT(trt_chkbx1(int)));

    connect(seismic_basket_add, SIGNAL(clicked()), this, SLOT(trt_seismic_basket_add()));
    connect(seismic_basket_sub, SIGNAL(clicked()), this, SLOT(trt_seismic_basket_sub()));
    connect(horizons_basket_add, SIGNAL(clicked()), this, SLOT(trt_horizons_basket_add()));
    connect(horizons_basket_sub, SIGNAL(clicked()), this, SLOT(trt_horizons_basket_sub()));
    connect(culturals_basket_add, SIGNAL(clicked()), this, SLOT(trt_culturals_basket_add()));
    connect(culturals_basket_sub, SIGNAL(clicked()), this, SLOT(trt_culturals_basket_sub()));

    connect(qpb_well_basket_add, SIGNAL(clicked()), this, SLOT(trt_well_basket_add()));
    connect(qpb_well_basket_sub, SIGNAL(clicked()), this, SLOT(trt_well_basket_sub()));
    connect(qpb_welllog_basket_add, SIGNAL(clicked()), this, SLOT(trt_welllog_basket_add()));
    connect(qpb_welllog_basket_sub, SIGNAL(clicked()), this, SLOT(trt_welllog_basket_sub()));
    connect(qpb_welltf2p_basket_add, SIGNAL(clicked()), this, SLOT(trt_welltf2p_basket_add()));
    connect(qpb_welltf2p_basket_sub, SIGNAL(clicked()), this, SLOT(trt_welltf2p_basket_sub()));
    connect(qpb_wellpicks_basket_add, SIGNAL(clicked()), this, SLOT(trt_wellpicks_basket_add()));
    connect(qpb_wellpicks_basket_sub, SIGNAL(clicked()), this, SLOT(trt_wellpicks_basket_sub()));

    connect(qpb_rgb_basket_add, SIGNAL(clicked()), this, SLOT(trt_rgb_basket_add()));
    connect(qpb_rgb_basket_sub, SIGNAL(clicked()), this, SLOT(trt_rgb_basket_sub()));

    connect(nurbs_basket_add, SIGNAL(clicked()), this, SLOT(trt_nurbs_basket_add()));
    connect(nurbs_basket_sub, SIGNAL(clicked()), this, SLOT(trt_nurbs_basket_sub()));

    connect(qpb_cultural_database_update, SIGNAL(clicked()), this, SLOT(trt_cultural_database_update()));
    connect(qpb_well_database_update, SIGNAL(clicked()), this, SLOT(trt_well_database_update()));
    connect(qpb_horizon_database_update, SIGNAL(clicked()), this, SLOT(trt_horizon_database_update()));
    connect(qpb_seismic_database_update, SIGNAL(clicked()), this, SLOT(trt_seismic_database_update()));
    connect(qpb_nurbs_database_update, SIGNAL(clicked()), this, SLOT(trt_nurbs_database_update()));

    // connect(lineedit_custompath, SIGNAL(returnPressed(QString)), this, SLOT(trt_lineedit_custompath_return()));
    connect(qpb_custompath, SIGNAL(clicked()), this, SLOT(trt_custompath_valid()));
    connect(qpb_debug, SIGNAL(clicked()), this, SLOT(trt_debug()));

    // auto load last session
    load_last_session();
}


void GeotimeProjectManagerWidget::fill_empty_logs_list()
{

//#pragma omp parallel for schedule(dynamic)
	for(int i=0;i<well_list.size();i++)
	{

		int idx_well = -1;
		int idx_bore = -1;

		for (int n=0; n<display0.wells.size(); n++)
		{
			if ( display0.wells[n].head_fullname.compare(well_list[i].head_fullname) == 0 )
			{
				idx_well = n;
				break;
			}
		}

		if(idx_well<0) continue;


		for(int j=0;j<well_list[i].wellborelist.size();j++)
		{
			for (int m=0; m<display0.wells[idx_well].bore.size(); m++)
			{

				if ( display0.wells[idx_well].bore[m].bore_fullname.compare(well_list[i].wellborelist[j].bore_fullname) == 0 )
				{
					idx_bore = m;
					break;
				}
			}

			if(idx_well != -1 && idx_bore != -1)
			{
				if(well_list[i].wellborelist[j].log_tinyname.size() == 0)
				{
					well_list[i].wellborelist[j].log_tinyname =  display0.wells[idx_well].bore[idx_bore].log_tinyname;
					well_list[i].wellborelist[j].log_fullname = display0.wells[idx_well].bore[idx_bore].log_fullname;
				}
			}
		}
	}



	display_welllog_basket_list();




}



GeotimeProjectManagerWidget::~GeotimeProjectManagerWidget()
{
    
}

void GeotimeProjectManagerWidget::setCulturalsVisible(bool val)
{

}

void GeotimeProjectManagerWidget::setNeuronsVisible(bool val)
{

}

void GeotimeProjectManagerWidget::setWellsVisible(bool val)
{

}

void GeotimeProjectManagerWidget::setHorizonsVisible(bool val)
{

}

void GeotimeProjectManagerWidget::setSeismicVisible(bool val)
{
    
}

void GeotimeProjectManagerWidget::memoryRAZ()
{
    well_list.clear();
    seismic_fullname.clear();
    seismic_tinyname.clear();
    m_seismic_fullpath.clear();
    seismic_diplay_color.clear();
    seismic_tinyname_basket.clear();
    m_seismic_fullpath_basket.clear();
    seismic_basket_color.clear();
    horizons_fullname.clear();
    horizons_tinyname.clear();
    horizons_tinyname_basket.clear();
    horizons_fullname_basket.clear();

    nurbs_fullname.clear();
    nurbs_tinyname.clear();

	nurbs_tinyname_basket.clear();
	nurbs_fullname_basket.clear();

	freehorizons_tinyname_basket.clear();
	freehorizons_fullname_basket.clear();
	isohorizons_tinyname_basket.clear();
	isohorizons_fullname_basket.clear();
	horizonanims_tinyname_basket.clear();
	horizonanims_fullname_basket.clear();
}

// public
// =============================================================
void GeotimeProjectManagerWidget::removeTab(QString name)
{
	int N = tabw_table1->count();
	for (int n=0; n<N; n++)
	{
		QString str = tabw_table1->tabText(n);
		if ( str.compare(name) == 0 )
		{
			tabw_table1->removeTab(n);
			return;
		}
	}
}

void GeotimeProjectManagerWidget::removeTabSeismic()
{
	removeTab(QString("Seismic"));
	 tab_seismic = false;
}

void GeotimeProjectManagerWidget::removeTabNurbs()
{
	removeTab(QString("Nurbs"));
	 tab_nurbs = false;
}

void GeotimeProjectManagerWidget::removeTabHorizons()
{
	removeTab(QString("Horizons"));
    tab_horizons = false;
}

void GeotimeProjectManagerWidget::removeTabCulturals()
{
	removeTab(QString("Culturals"));
    tab_culturals = false;
}

void GeotimeProjectManagerWidget::removeTabWells()
{
	removeTab(QString("Wells / wellbores"));
    tab_wells = false;
}

void GeotimeProjectManagerWidget::removeTabNeurons()
{
	removeTab(QString("Neurons"));
    tab_neurons = false;
}

void GeotimeProjectManagerWidget::removeTabPicks()
{
	removeTab(QString("Picks"));
    tab_picks = false;
}

bool GeotimeProjectManagerWidget::isTabSeismic()
{
	return tab_seismic;
}

bool GeotimeProjectManagerWidget::isTabNurbs()
{
	return tab_nurbs;
}

bool GeotimeProjectManagerWidget::isTabHorizons()
{
	return tab_horizons;
}

bool GeotimeProjectManagerWidget::isTabCulturals()
{
	return tab_culturals;
}

bool GeotimeProjectManagerWidget::isTabWells()
{
	return tab_wells;
}

bool GeotimeProjectManagerWidget::isTabNeurons()
{
	return tab_neurons;
}

bool GeotimeProjectManagerWidget::isTabPicks()
{
	return tab_picks;
}


QString GeotimeProjectManagerWidget::get_projet_name()
{
    if ( lw_projetlist->currentItem() == NULL ) return QString(".");
    QString project_name = lw_projetlist->currentItem()->text();  
    return project_name;
}

QString GeotimeProjectManagerWidget::get_projet_fullpath_name()
{
    if ( lw_projetlist->currentItem() == NULL ) return QString(".");
    QString project_name = lw_projetlist->currentItem()->text(); 
    return get_project_path0() + project_name + "/";
}

QString GeotimeProjectManagerWidget::get_survey_name()
{
    if ( lw_surveylist->currentItem() == NULL ) return QString(".");
    QString name = lw_surveylist->currentItem()->text(); 
    return name;
}

QString GeotimeProjectManagerWidget::get_survey_fullpath_name()
{
    if ( lw_surveylist->currentItem() == NULL ) return QString(".");
    QString name = lw_surveylist->currentItem()->text();
    int idx = getIndexFromVectorString(survey_name, name);
    if ( idx >= 0 )
    {
        name = survey_dirname[idx];
    }
    return get_survey_path0() + name + "/";
}

QString GeotimeProjectManagerWidget::getNextVisionPath()
{
	return get_survey_fullpath_name() + "/" + QString::fromStdString(GeotimePath::NEXTVISION_IMPORT_EXPORT_DIR) + "/" + QString::fromStdString(GeotimePath::NEXTVISION_MAIN_DIR) + "/";
}

QString GeotimeProjectManagerWidget::getNVHorizonPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_DIR) + "/";
}

QString GeotimeProjectManagerWidget::getIsoHorizonPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_ISOHORIZON_DIR) + "/";
}

QString GeotimeProjectManagerWidget::getNextVisionSeismicPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_SEISMIC_DIR) + "/";
}

QString GeotimeProjectManagerWidget::getPatchPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_PATCH_DIR) + "/";
}

QString GeotimeProjectManagerWidget::getVideoPath()
{
	return getNextVisionPath() + QString::fromStdString(GeotimePath::NEXTVISION_VIDEO_DIR) + "/";
}


std::vector<QString> GeotimeProjectManagerWidget::get_seismic_names()
{
    return this->seismic_tinyname_basket;
}

std::vector<QString> GeotimeProjectManagerWidget::get_nurbs_names()
{
	 return this->nurbs_tinyname_basket;

}

std::vector<QString> GeotimeProjectManagerWidget::get_seismic_AllTinynames()
{
    return this->seismic_tinyname;
}

std::vector<QString> GeotimeProjectManagerWidget::get_seismic_AllFullnames()
{
	/*
	std::vector<QString> names;
	QString path = get_seismic_path0();
	names.resize(seismic_fullname.size());
	for (int i=0; i<seismic_fullname.size(); i++)
	{
		names[i] = path + seismic_fullname[i];
	}
	return names;*/
	return m_seismic_fullpath;
}

//std::vector<QString> GeotimeProjectManagerWidget::get_seismic_fullnames()
//{
//    return this->seismic_fullname_basket;
//}

std::vector<QString> GeotimeProjectManagerWidget::get_nurbs_fullnames()
{
    return this->nurbs_fullname_basket;
}

std::vector<QString> GeotimeProjectManagerWidget::get_seismic_fullpath_names()
{
	/*
    std::vector<QString> names;
    QString path = get_seismic_path0();
    // QList<QListWidgetItem*> list0 = lw_seismic->selectedItems();
    names.resize(seismic_fullname_basket.size());
    for (int i=0; i<seismic_fullname_basket.size(); i++)
    {
        names[i] = path + seismic_fullname_basket[i];
    }        
    return names;
    */
	return m_seismic_fullpath_basket;
}


std::vector<QString> GeotimeProjectManagerWidget::get_horizon_names()
{
   return this->horizons_tinyname_basket;
}

std::vector<QString> GeotimeProjectManagerWidget::get_horizon_fullnames()
{
   return this->horizons_fullname_basket;
}


std::vector<QString> GeotimeProjectManagerWidget::get_horizon_fullpath_names()
{
   return this->horizons_fullname_basket;
}


std::vector<QString> GeotimeProjectManagerWidget::get_rgb_names()
{
    return this->rgb_basket_tinyname;
}

std::vector<QString> GeotimeProjectManagerWidget::get_rgb_fullnames()
{
    return this->rgb_basket_fullname;
}



std::vector<WELLLIST> GeotimeProjectManagerWidget::get_well_list()
{
	return this->well_list;
}

std::vector<PMANAGER_WELL_DISPLAY> GeotimeProjectManagerWidget::get_display_well_list()
{
	return this->display0.wells;
}

std::vector<QString> GeotimeProjectManagerWidget::get_culturals_cdat_names()
{
	return data0.culturals.cdata_tinyname;
}

std::vector<QString> GeotimeProjectManagerWidget::get_culturals_cdat_fullnames()
{
	return data0.culturals.cdata_fullname;
}

std::vector<QString> GeotimeProjectManagerWidget::get_culturals_strd_names()
{
	return data0.culturals.strd_tinyname;
}

std::vector<QString> GeotimeProjectManagerWidget::get_culturals_strd_fullnames()
{
	return data0.culturals.strd_fullname;
}

std::vector<QString> GeotimeProjectManagerWidget::get_wells_names()
{
    std::vector<QString> names;
    QList<QListWidgetItem*> list0 = lw_wells->selectedItems();
    names.resize(list0.size());
    for (int i=0; i<list0.size(); i++)
        names[i] = list0[i]->text();
    return names;
}

std::vector<QString> GeotimeProjectManagerWidget::get_wells_fullnames()
{
    std::vector<QString> names;
    QList<QListWidgetItem*> list0 = lw_wells->selectedItems();
    names.resize(list0.size());
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(wells_tinyname, txt);
        if ( idx >= 0 )
        {
            names[i] = wells_fullname[idx];
        }
    }        
    return names;
}

std::vector<QString> GeotimeProjectManagerWidget::get_wells_fullpath_names()
{
    std::vector<QString> names;
    QString path = get_wells_path0();
    QList<QListWidgetItem*> list0 = lw_wells->selectedItems();
    names.resize(list0.size());
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(wells_tinyname, txt);
        if ( idx >= 0 )
        {
            names[i] = path + wells_fullname[idx];
        }
    }        
    return names;
}

bool GeotimeProjectManagerWidget::select_well_fullpath_name(const QString& searched_fullpath_name)
{
	bool found = false;
	long i=0;
	while (!found && i<lw_wells->count())
	{
		QListWidgetItem* item = lw_wells->item(i);
		QString fullpath_name = item->data(Qt::UserRole).toString();

		found = searched_fullpath_name.compare(fullpath_name)==0;
		if (!found)
		{
			i++;
		}
	}
	if (found)
	{
		QListWidgetItem* item = lw_wells->item(i);
		item->setSelected(true);
	}
	return found;
}

void GeotimeProjectManagerWidget::clear_well_gui_selection()
{
	lw_wells->clearSelection();
}

bool GeotimeProjectManagerWidget::select_well_basket_fullpath_name(const QString& searched_fullpath_name)
{
	bool found = false;
	long i=0;
    QString path = get_wells_path0();
	while (!found && i<lw_wellsbasket->count())
	{
		QListWidgetItem* item = lw_wellsbasket->item(i);
		QString tinyname = item->text();
		int idx = getIndexFromVectorString(well_wellbore_basket, tinyname);
		QString fullpath_name;
		if (idx>=0)
		{
			fullpath_name = well_bore_fullname_basket[idx];
		}

		found = searched_fullpath_name.compare(fullpath_name)==0;
		if (!found)
		{
			i++;
		}
	}
	if (found)
	{
		QListWidgetItem* item = lw_wellsbasket->item(i);
		item->setSelected(true);
	}
	return found;
}

void GeotimeProjectManagerWidget::clear_well_basket_gui_selection()
{
	lw_wellsbasket->clearSelection();
}

std::vector<QString> GeotimeProjectManagerWidget::get_neurons_names()
{
    std::vector<QString> names;
    QList<QListWidgetItem*> list0 = lw_neurons->selectedItems();
    names.resize(list0.size());
    for (int i=0; i<list0.size(); i++)
        names[i] = list0[i]->text();
    return names;
}

std::vector<QString> GeotimeProjectManagerWidget::get_neurons_fullnames()
{
    std::vector<QString> names;
    QList<QListWidgetItem*> list0 = lw_neurons->selectedItems();
    names.resize(list0.size());
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(neurons_tinyname, txt);
        if ( idx >= 0 )
        {
            names[i] = neurons_fullname[idx];
        }
    }        
    return names;
}

std::vector<QString> GeotimeProjectManagerWidget::get_neurons_fullpath_names()
{
    std::vector<QString> names;
    QString path = get_neurons_path0();
    QList<QListWidgetItem*> list0 = lw_neurons->selectedItems();
    names.resize(list0.size());
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(neurons_tinyname, txt);
        if ( idx >= 0 )
        {
            names[i] = path + neurons_fullname[idx];
        }
    }        
    return names;
}

QString GeotimeProjectManagerWidget::getProjIndexNameForDataBase()
{
	int idx = this->cb_projecttype->currentIndex();
	if ( idx < this->cb_projecttype->count()-1 )
	{
		return QString::number(idx);
	}
	else
	{
		QString pathTmp = lineedit_custompath->text();
		pathTmp.replace("/", "_");
		QString ret = QString::number(idx) + pathTmp ;
		return ret;
	}
}

QString GeotimeProjectManagerWidget::get_well_database_name()
{
	// int idx = this->cb_projecttype->currentIndex();
	GlobalConfig& config = GlobalConfig::getConfig();
	QString name = get_wells_path0(); // should not have redondant / and path should end with a /
	name.replace("/", "_@_");
	QString ret = config.databasePath() + QString("/database_wells_") + name + ".txt";
	return ret;
}


QString GeotimeProjectManagerWidget::get_cultural_database_name()
{
	GlobalConfig& config = GlobalConfig::getConfig();
	int idx = this->cb_projecttype->currentIndex();
	QString proj = this->lw_projetlist->currentItem()->text();
	QString ret = config.databasePath() + "/database_culturals_" + getProjIndexNameForDataBase() + QString("_") + proj + ".txt";
	return ret;
}

QString GeotimeProjectManagerWidget::get_seismic_database_name()
{
	GlobalConfig& config = GlobalConfig::getConfig();
	QString name = get_seismic_path0(); // should not have redondant / and path should end with a /
	name.replace("/", "_@_");
	QString ret = config.databasePath() + QString("/databasev2_seismic_") + name + ".txt";
	return ret;

}

QString GeotimeProjectManagerWidget::get_horizons_database_name()
{
	GlobalConfig& config = GlobalConfig::getConfig();
	QString name = get_horizons_path0(); // should not have redondant / and path should end with a /
	name.replace("/", "_@_");
	QString ret = config.databasePath() + QString("/database_horizon_") + name + ".txt";
	return ret;
}



// new version
/*
void GeotimeProjectManagerWidget::clear_data()
{
	data0.project_type = 0;
}

void GeotimeProjectManagerWidget::clear_display()
{

}
*/

void GeotimeProjectManagerWidget::getIndexFromWellWellboreFull(QString txt, int *idx_well, int *idx_bore)
{
	*idx_well = -1;
	*idx_bore = -1;

	for (int iw=0; iw<display0.wells.size(); iw++)
	{
		for (int ib=0; ib<display0.wells[iw].bore.size(); ib++)
		{
			QString str0 = display0.wells[iw].bore[ib].bore_fullname;
			if ( str0.compare(txt) == 0 )
			{
				*idx_well = iw;
				*idx_bore = ib;
				return;
			}
		}
	}
}




int GeotimeProjectManagerWidget::getIndexFromVectorString(std::vector<QString> list, QString txt)
{
    for (int i=0; i<list.size(); i++)
    {
        if ( list[i].compare(txt) == 0 )
            return i;
    }
    return -1;
}

std::vector<std::vector<QString>> GeotimeProjectManagerWidget::multiCriterionSearchFormat(QString str)
{
	QStringList list0 = str.split(";", Qt::SkipEmptyParts);//QString::SkipEmptyParts);

	std::vector<std::vector<QString>> listf;
	listf.resize(4);

	for (int n=0; n<list0.size(); n++)
	{
		QStringList list = list0[n].split("=", Qt::SkipEmptyParts);
		if ( list.size() == 1 )
		{
			listf[0].push_back(list[0]);
		}
		else
		{
			QStringList listx = list[1].split(" ", Qt::SkipEmptyParts);
			if ( list[0].compare("log") == 0  )
			{
				for (int p=0; p<listx.size(); p++)
				{
					listf[1].push_back(listx[p]);
				}
			}
			else if ( list[0].compare("tf2p") == 0 )
			{
				for (int p=0; p<listx.size(); p++)
				{
					listf[2].push_back(listx[p]);
				}
			}
			else if ( list[0].compare("picks") == 0 )
			{
				for (int p=0; p<listx.size(); p++)
				{
					listf[3].push_back(listx[p]);
				}
			}
		}
	}
	return listf;
}


int GeotimeProjectManagerWidget::qstring_display_valid(QString str, QString prefix)
{
	QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
	int nbsearch = list1.size();
	if ( nbsearch == 0 ) return 1;
	int val = 0;
	for ( int s=0; s<nbsearch; s++)
	{
		int idx = str.indexOf(list1[s], 0, Qt::CaseInsensitive);
	    if ( idx >= 0 || prefix.isEmpty() ) val ++;
	}
	// if ( val == nbsearch || nbsearch == 0 ) return 1;
	if ( val > 0 ) return 1;
	return 0;
}

int GeotimeProjectManagerWidget::qstring_display_listPrefix_valid(QString str, std::vector<QString> prefix)
{
	int nbsearch = prefix.size();
	if ( nbsearch == 0 ) return 1;
	int val = 0;
	for ( int s=0; s<nbsearch; s++)
	{
		int idx = str.indexOf(prefix[s], 0, Qt::CaseInsensitive);
	    if ( idx >= 0 ) val ++;
	}
	// if ( val == nbsearch || nbsearch == 0 ) return 1;
	if ( val > 0 ) return 1;
	return 0;
}


/*
int GeotimeProjectManagerWidget::filext_axis(QString filename)
{
	// fprintf(stderr, "-> %s\n", filename.toStdString().c_str());
	QFile file(filename);
	size_t size = file.size();
	if ( size < 5120 )
	{
		fprintf(stderr, "error reading axis on file %s\n", filename.toStdString().c_str());
		return 0;
	}	inri::Xt *_xt;
	_xt = new inri::Xt(filename.toStdString().c_str());
	int axis = 0;
	try
	{
		axis = _xt->axis();
		// throw string("Division par zéro !");
		delete _xt;
		return axis;
	}
	catch (const std::exception& e)
	{
		fprintf(stderr, "error xt get axis: %s --> %s\n", filename.toStdString().c_str(), e.what());
		delete _xt;
		return 0;
	}
}
*/

int GeotimeProjectManagerWidget::filext_axis(QString filename)
{
	QProcess process;
	QStringList options;
	options << filename;
	process.start("TestXtFile", options);
	process.waitForFinished();

	if (process.exitCode()!=QProcess::NormalExit) {
		std::cerr << "provided file is not in xt format (" << filename.toStdString() << ")" << std::endl;
		return 2;
	}

	// fprintf(stderr, "-> %s\n", filename.toStdString().c_str());
	std::size_t offset;
	{
		inri::Xt xt(filename.toStdString().c_str());
		if (!xt.is_valid()) {
			std::cerr << "xt cube is not valid (" << filename.toStdString() << ")" << std::endl;
			return 2;
		}
		offset = (size_t)xt.header_size();
	}
	QFile file(filename);
	size_t size = file.size();
	if ( size < offset )
	{
		fprintf(stderr, "error reading axis on file %s\n", filename.toStdString().c_str());
		return 2;
	}
	FILE *pFile = fopen(filename.toStdString().c_str(), "r");
	if ( pFile == NULL ) return 0;
	char str[offset];
	fseek(pFile, 0x4c, SEEK_SET);
	int n = 0, cont = 1;
	int typeAxe1 = -1;
	while ( cont )
	{
		int nbre = fscanf(pFile, "TYPE_AXE1=\t%d\n", &typeAxe1);
		if ( nbre > 0 )
			cont = 0;
		else
			fgets(str, offset, pFile);
		n++;
		if ( n > 20 )
		{
			cont = 0;
			strcpy(str, "Other");
		}
	}
	fclose(pFile);
//	if ( strcmp(str, "Time") == 0 ) return 0;
//	if ( strcmp(str, "Depth") == 0 ) return 1;
	if (typeAxe1==1) return 0;
	if (typeAxe1==2) return 1;
	return 2;
}


QString GeotimeProjectManagerWidget::get_project_path(int project_type)
{
    QString ret = "";
    GlobalConfig& config = GlobalConfig::getConfig();
    const std::vector<std::pair<QString, QString>>& dir_projects = config.dirProjects();
    if (project_type<=0) {
        ret = "";
    } else if (project_type-1<dir_projects.size()) {
    	ret = dir_projects[project_type-1].second;
    } else if (project_type-1==dir_projects.size()) {
    	ret = lineedit_custompath->text() + "/";
    } else {
        ret = "";
    }
    return ret;
}


QString GeotimeProjectManagerWidget::get_survey_subdirectory()
{
	return QString("DATA/3D");
}

std::vector<QString> GeotimeProjectManagerWidget::get_directories(QString path)
{
    std::vector<QString> tab;
    FILE *pfile;

    char buff[10000], b[10000];
    fprintf(stderr, "path: %s\n",  path.toStdString().c_str());
    sprintf(buff, "find %s -mindepth 1 -maxdepth 1 -exec basename \\{} \\; | sort", path.toStdString().c_str());
    if ( ( pfile = popen(buff, "r") ) == NULL )
    {
        fprintf (stderr, "erreur");
        return tab;
    }
    while (fgets (buff, sizeof(buff), pfile) != NULL)
    {
        strtok(buff, "\n");
        strcpy(b, buff);
        if ( buff[0] != 0 && strcmp(buff, ".") != 0 ) tab.push_back(QString(b));
    }
    fclose(pfile);
    return tab;
}

void GeotimeProjectManagerWidget::project_list_init()
{
	this->display0.project_main_list.resize(cb_projecttype->count());
    for (int i=0; i<display0.project_main_list.size()-1; i++)
    {
        QString path = get_project_path(i+1);
        this->display0.project_main_list[i+1].tinyname = get_directories(path);

        int N = this->display0.project_main_list[i+1].tinyname.size();
        this->display0.project_main_list[i+1].fullname.resize(N);
        for (int n=0; n<N; n++)
        {
        	this->display0.project_main_list[i+1].fullname[n] = path + this->display0.project_main_list[i+1].tinyname[n];
        }
    }
}

void GeotimeProjectManagerWidget::survey_names_update()
{
	/*
	int idx = getIndexFromVectorString(display0.project_main_list[display0.project_type].tinyname, lw_projetlist->currentItem()->text());
	if ( idx < 0 ) return;
	data0.project.tinyname = display0.project_main_list[display0.project_type].tinyname[idx];
	data0.project.fullname = display0.project_main_list[display0.project_type].fullname[idx];

	qDebug() << data0.project.tinyname;
	qDebug() << data0.project.fullname;
	*/


	// data0.project =
    char buff[10000];
    QString survey_path = get_survey_path0();// qDebug() << survey_path;
    QFileInfoList list = get_dirlist(survey_path);
    int N = list.size();
    survey_dirname.resize(N);
    survey_name.resize(N);

    for (int i=0; i<N; i++)
    {
        QFileInfo fileInfo = list.at(i);
        QString filename = fileInfo.fileName();
        QString desc_filename = survey_path + filename + "/" + filename + ".desc";
        FILE *pfile = fopen(desc_filename.toStdString().c_str(), "r");
        if ( pfile != NULL )
        {
            fgets(buff, 10000, pfile);
            fgets(buff, 10000, pfile);
            fgets(buff, 10000, pfile);
            fscanf(pfile, "name=%s\n", buff);
            fclose(pfile);
            survey_name[i] = QString(buff);
        }
        else
        {
            survey_name[i] = filename;
        }
        survey_dirname[i] = filename;
    }
}


void GeotimeProjectManagerWidget::trt_lineedit_custompath_return()
{
	qDebug() << "Return";
}

void GeotimeProjectManagerWidget::trt_custompath_valid()
{
	int idx = cb_projecttype->count()-1;

	// lineedit_custompath->setEnabled(false);
	// label_custom_path->setEnabled(false);

    this->lw_surveylist->clear();
    this->lw_projetlist->clear();
    this->lw_cultural->clear();
    this->lw_wells->clear();
    this->lw_neurons->clear();
    this->lw_horizons->clear();
    this->lw_seismic->clear();

    // this->lw_culturals->clear();
    data0.culturals.cdata_tinyname.clear();
    data0.culturals.cdata_fullname.clear();
    data0.culturals.strd_tinyname.clear();
    data0.culturals.strd_fullname.clear();
    display_cultural_list(lineedit_wellssearch->text());

    display_project_list(idx, lineedit_projectsearch->text());
    display_label_titles();
    clear_wells_data(1);
}


void GeotimeProjectManagerWidget::trt_projecttypeclick(int idx)
{
	if ( cb_projecttype->itemText(idx).compare("USER")==0 )
	{
		lineedit_custompath->setEnabled(true);
		label_custom_path->setEnabled(true);
		qpb_custompath->setEnabled(true);
	}
	else
	{
		lineedit_custompath->setEnabled(false);
		label_custom_path->setEnabled(false);
		qpb_custompath->setEnabled(false);
	}
    fprintf(stderr, "main project: %d\n", idx);
    this->lw_surveylist->clear();
    this->lw_projetlist->clear();
    this->lw_cultural->clear();
    this->lw_wells->clear();
    this->lw_neurons->clear();
    this->lw_horizons->clear();
    this->lw_seismic->clear();

    clear_project_specific_lists();

    // this->lw_culturals->clear();
    data0.culturals.cdata_tinyname.clear();
    data0.culturals.cdata_fullname.clear();
    data0.culturals.strd_tinyname.clear();
    data0.culturals.strd_fullname.clear();
    display_cultural_list(lineedit_wellssearch->text());

    display_project_list(idx, lineedit_projectsearch->text());

    display_label_titles();
    clear_wells_data(1);

    // this->projectpath = get_main_project_directory(idx);
    // update_project_list(project_list[idx], lineedit_projectsearch->text());
    // update_project_list(this->projectpath);

    /*
    // new version
    int idx0 = cb_projecttype->currentIndex();
    data0.project_type = idx0;
    display0.project_type = idx0;
    project_list_display(lineedit_projectsearch->text());
    */
}

void GeotimeProjectManagerWidget::trt_projetlistClick(QListWidgetItem* p)
{
    QString txt = p->text();
    // int idx = cb_projecttype->currentIndex();
    this->lw_cultural->clear();
    this->lw_wells->clear();
    this->lw_neurons->clear();
    this->lw_horizons->clear();
    this->lw_seismic->clear();
    this->well_list.clear();
    this->well_wellbore_basket.clear();
    clear_wells_data(1);

    clear_project_specific_lists();

    survey_names_update();
    display_survey_list(lineedit_surveysearch->text());

    wells_names_update();
    display_wells_list(lineedit_wellssearch->text());

    cultural_names_update();
    display_cultural_list(lineedit_wellssearch->text());
    neurons_names_update();
    display_neurons_list(lineedit_neuronssearch->text());
    display_label_titles();

    if ( m_picksManager )
    {
    	int idx = this->cb_projecttype->currentIndex();
    	QString name = get_projet_name();
    	QString path = get_projet_fullpath_name();
    	m_picksManager->setProjectType(idx);
    	m_picksManager->setProjectName(name);
    	m_picksManager->setProjectPath(path, name);
    }
}


void GeotimeProjectManagerWidget::project_list_display(QString prefix)
{

	int idx = cb_projecttype->currentIndex();
    this->lw_projetlist->clear();

    for (int i=0; i<this->display0.project_main_list[idx].tinyname.size(); i++)
    {
        QString str = display0.project_main_list[idx].tinyname[i];
        if ( qstring_display_valid(str, prefix ) )
        	this->lw_projetlist->addItem(str);
    }

/*
	int idx = cb_projecttype->currentIndex();
	this->lw_projetlist->clear();

	for (int i=0; i<this->display0.project_main_list[idx].tinyname.size(); i++)
	{
		QString str = display0.project_main_list[idx].tinyname[i];
	    if ( qstring_display_valid(str, prefix ) )
	    {
	    	QListWidgetItem *item = new QListWidgetItem;
	    	item->setText(str);
	    	item->setToolTip(str);
	    	item->setStatusTip(str);
	    	item->setWhatsThis(str);
	    	this->lw_projetlist->addItem(item);
	    }
	}
	*/
}

// =============================================================


QFileInfoList GeotimeProjectManagerWidget::get_seismic_list(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Files);
    // dir->setFilter(QDir::AllEntries );
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.xt" << "*.cwt";
    dir.setNameFilters(filters);
    QFileInfoList list = getFiles(path.toStdString(),filters); //dir.entryInfoList();
    return list;    
}

std::vector<QString> GeotimeProjectManagerWidget::get_seismic_list2(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Files);
    // dir->setFilter(QDir::AllEntries );
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.xt" << "*.cwt";
    dir.setNameFilters(filters);
    QFileInfoList list =getFiles(path.toStdString(),filters);// dir.entryInfoList();

    int N = list.size();
    std::vector<QString> seismic_list;
    seismic_list.resize(N);
    for (int i=0; i<list.size(); i++)
    {
        QFileInfo fileInfo = list.at(i);
        QString filename = fileInfo.fileName();
        seismic_list[i] = filename;
        // fprintf(stderr, "name: %s -> %s\n", path.toStdString().c_str(), filename.toStdString().c_str());
    }    
    return seismic_list;
}

std::vector<QString> GeotimeProjectManagerWidget::get_rgb_list(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Files);
    // dir->setFilter(QDir::AllEntries );
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.raw" << "*.xt";
    dir.setNameFilters(filters);
    QFileInfoList list =getFiles(path.toStdString(),filters);// dir.entryInfoList();

    int N = list.size();
    std::vector<QString> rgb_list;
    rgb_list.resize(N);
    for (int i=0; i<list.size(); i++)
    {
        QFileInfo fileInfo = list.at(i);
        QString filename = fileInfo.fileName();
        rgb_list[i] = filename;
        // fprintf(stderr, "name: %s -> %s\n", path.toStdString().c_str(), filename.toStdString().c_str());
    }
    return rgb_list;
}


QFileInfoList GeotimeProjectManagerWidget::get_horizons_list(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.iso" << "*.raw";
    dir.setNameFilters(filters);
    QFileInfoList list = getFiles(path.toStdString(),filters);//dir.entryInfoList();
    return list;    
}

/*
QFileInfoList GeotimeProjectManagerWidget::get_cultural_list(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.str";
    dir.setNameFilters(filters);
    QFileInfoList list = dir.entryInfoList();
    return list;    
}
*/


QFileInfoList GeotimeProjectManagerWidget::get_neurons_list(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
    dir.setSorting(QDir::Name);
    QFileInfoList list =dir.entryInfoList();
    return list;    
}

std::vector<QString> GeotimeProjectManagerWidget::get_nurbs_list(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Files);
    // dir->setFilter(QDir::AllEntries );
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.txt";// << "*.cwt";
    dir.setNameFilters(filters);
    QFileInfoList list = getFiles(path.toStdString(),filters);//dir.entryInfoList();

    int N = list.size();


    std::vector<QString> nurbs_list;
    nurbs_list.resize(N);
    for (int i=0; i<list.size(); i++)
    {
        QFileInfo fileInfo = list.at(i);
        QString filename = fileInfo.fileName();
        nurbs_list[i] = filename;
        // fprintf(stderr, "name: %s -> %s\n", path.toStdString().c_str(), filename.toStdString().c_str());
    }
    return nurbs_list;
}

std::vector<QString> GeotimeProjectManagerWidget::get_horizonanim_list(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Files);
    // dir->setFilter(QDir::AllEntries );
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.hor";// << "*.cwt";
    dir.setNameFilters(filters);
    QFileInfoList list = getFiles(path.toStdString(),filters);//dir.entryInfoList();
    int N = list.size();
    std::vector<QString> horanim_list;
    horanim_list.resize(N);
    for (int i=0; i<list.size(); i++)
    {
        QFileInfo fileInfo = list.at(i);
        QString filename = fileInfo.fileName();
        horanim_list[i] = filename;
        // fprintf(stderr, "name: %s -> %s\n", path.toStdString().c_str(), filename.toStdString().c_str());
    }
    return horanim_list;
}

QString GeotimeProjectManagerWidget::get_survey_path(int project_type, QString project_name)
{
     return get_project_path(project_type) + project_name + "/DATA/3D/";
}

QString GeotimeProjectManagerWidget::get_seismic_path(int project_type, QString project_name, QString survey_name)
{
    return get_survey_path(project_type, project_name) + survey_name + "/DATA/SEISMIC/";
}

QString GeotimeProjectManagerWidget::get_horizons_path(int project_type, QString project_name, QString survey_name)
{
    return get_survey_path(project_type, project_name) + survey_name + "/DATA/HORIZONS/";
}

QString GeotimeProjectManagerWidget::get_wells_path(int project_type, QString project_name)
{
    return get_project_path(project_type) + project_name + "/DATA/WELLS/";
}

QString GeotimeProjectManagerWidget::get_cultural_path(int project_type, QString project_name)
{
    return get_project_path(project_type) + project_name + "/DATA/WELLS/";
}

QString GeotimeProjectManagerWidget::get_nurbs_path0()
{
    return get_IJKPath() + "GraphicLayers/Nurbs/";
}

QString GeotimeProjectManagerWidget::get_horizonanim_path0()
{
    return get_IJKPath() + "HORIZONS/" + QString::fromStdString(FreeHorizonManager::BaseDirectory) + "/Animations/";
}

QString GeotimeProjectManagerWidget::get_project_path0()
{
    int idx = this->cb_projecttype->currentIndex();
    QString ret = get_project_path(idx);
    return ret;
}

QString GeotimeProjectManagerWidget::get_survey_path0()
{
    if ( this->lw_projetlist->currentItem() == NULL ) return QString(".");
    QString project_path = get_project_path0();
    QString project_name = this->lw_projetlist->currentItem()->text();
    return project_path + project_name + "/DATA/3D/";
}

QString GeotimeProjectManagerWidget::get_seismic_path0()
{
    if ( this->lw_surveylist->currentItem() == NULL ) return QString(".");
    QString survey_tiny_name = this->lw_surveylist->currentItem()->text();
    int idx = getIndexFromVectorString(this->survey_name, survey_tiny_name);
    if ( idx == -1 ) return QString("");
    QString survey_full_name = this->survey_dirname[idx];

    QString seismic_path = get_survey_path0() + survey_full_name + "/DATA/SEISMIC/";

    // remove redundant /
    seismic_path = format_dir_path(seismic_path);
    return seismic_path;
}

QString GeotimeProjectManagerWidget::get_ImportExportPath()
{
    if ( this->lw_surveylist->currentItem() == NULL ) return QString(".");
    QString survey_tiny_name = this->lw_surveylist->currentItem()->text();
    int idx = getIndexFromVectorString(this->survey_name, survey_tiny_name);
    if ( idx == -1 ) return QString("");
    QString survey_full_name = this->survey_dirname[idx];
    return get_survey_path0() + survey_full_name + "/ImportExport/";
}

QString GeotimeProjectManagerWidget::get_IJKPath()
{
	return get_ImportExportPath() + "IJK/";
}


QString GeotimeProjectManagerWidget::get_cubeRGT2RGBPath()
{
	return get_ImportExportPath() + "IJK/cubeRGT2RGB/";
}

QString GeotimeProjectManagerWidget::get_horizons_path0()
{
    if ( this->lw_surveylist->currentItem() == NULL ) return QString(".");
    QString survey_tiny_name = this->lw_surveylist->currentItem()->text();
    int idx = getIndexFromVectorString(this->survey_name, survey_tiny_name);
    if ( idx == -1 ) return QString("");
    QString survey_full_name = this->survey_dirname[idx];
    // return get_survey_path0() + survey_full_name + "/DATA/HORIZONS/";
    QString horizons_path = get_survey_path0() + survey_full_name + "/ImportExport/IJK/";

    // remove redondant /
    horizons_path = format_dir_path(horizons_path);
    return horizons_path;
}

QString GeotimeProjectManagerWidget::get_wells_path0()
{
    if ( this->lw_projetlist->currentItem() == NULL ) return QString(".");
    QString project_path = get_project_path0();
    QString project_name = this->lw_projetlist->currentItem()->text();

    QString wells_path = project_path + project_name + "/DATA/WELLS/";

    // remove redondant /
    wells_path = format_dir_path(wells_path);
    return wells_path;
}

QString GeotimeProjectManagerWidget::get_cultural_path0()
{
    if ( this->lw_projetlist->currentItem() == NULL ) return QString(".");
    QString project_path = get_project_path0();
    QString project_name = this->lw_projetlist->currentItem()->text();
    QString cultural_path = project_path + project_name + "/DATA/CULTURAL/";

    // remove redondant /
    cultural_path = format_dir_path(cultural_path);
    return cultural_path;
}

QString GeotimeProjectManagerWidget::get_neurons_path0()
{
    if ( this->lw_projetlist->currentItem() == NULL ) return QString(".");
    QString project_path = get_project_path0();
    QString project_name = this->lw_projetlist->currentItem()->text();
    // return project_path + project_name + "/DATA/NEURONS/";

    QString neurons_path = project_path + project_name + "/DATA/NEURONS/neurons2/LogInversion2Problem3/";

    // remove redondant /
    neurons_path = format_dir_path(neurons_path);
    return neurons_path;
}




// ==================================================
void GeotimeProjectManagerWidget::display_label_titles()
{
    QString project = QString("..."), survey = QString("...");
    QListWidgetItem *p = this->lw_projetlist->currentItem();
    if ( p != NULL ) project = p->text();
    p = this->lw_surveylist->currentItem();
    if ( p != NULL ) survey = p->text();    
    // this->label_projectname->setText(project + QString(" - ") + survey);
    this->qgb_projectmanager->setTitle(project + QString(" - ") + survey);
}

void GeotimeProjectManagerWidget::display_project_list(int project_type, QString prefix)
{
	if ( project_list[project_type].size() == 0 && project_type < cb_projecttype->count()-1 )
	{
		QString path = get_project_path(project_type);
		this->project_list[project_type] = get_dirlist2(path);
	}
	if ( project_type == cb_projecttype->count()-1 )
	{
		QString path = get_project_path(project_type);
		this->project_list[project_type] = get_dirlist2(path);
	}
    std::vector<QString> tab = project_list[project_type];
    this->lw_projetlist->clear();
    QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
    int nbsearch = list1.size();
    // for (int i=0; i<nbsearch; i++) fprintf(stderr, "----> %d %s\n", i, list1[i].toStdString().c_str());

    for (int i=0; i<tab.size(); i++)
    {
        // char *txt = tab.at(i);
        // QString str = QString(txt);
    	QString str = tab[i];

        int val = 0;
        for ( int s=0; s<nbsearch; s++)
        {
            int idx = str.indexOf(list1[s], 0, Qt::CaseInsensitive);
            if ( idx >= 0 || prefix.isEmpty() ) val ++;
        }
       if ( val == nbsearch || nbsearch == 0)
       {
    	   QListWidgetItem *item = new QListWidgetItem;
    	   item->setText(str);
    	   item->setToolTip(str);
    	   // item->setStatusTip(QString(txt));
    	   // item->setWhatsThis(QString(txt));
    	   this->lw_projetlist->addItem(item);
       }
    }
}

void GeotimeProjectManagerWidget::display_survey_list(QString prefix)
{
    this->lw_surveylist->clear();
    QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
    int nbsearch = list1.size();   

    for (int i=0; i<this->survey_name.size(); i++)
    {
        int val = 0;
        for (int s=0; s<nbsearch; s++)
        {
            int idx = this->survey_name[i].indexOf(list1[s], 0, Qt::CaseInsensitive);
            if ( idx >=0 || prefix.isEmpty() ) val++;
        }        
        if ( val == nbsearch || nbsearch == 0)
        {
        	QListWidgetItem *item = new QListWidgetItem;
        	item->setText(survey_name[i]);
        	item->setToolTip(survey_name[i]);
        	this->lw_surveylist->addItem(item);
        }
    }
}
 
void GeotimeProjectManagerWidget::display_seismic_list(QString prefix)
{
    this->lw_seismic->clear();
    QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
    int nbsearch = list1.size(); 

    // int cpt = 0;
    for (int i=0; i<this->seismic_tinyname.size(); i++)
    {
         // int idx = seismic_tinyname[i].indexOf(prefix, 0, Qt::CaseInsensitive);

         int val = 0;
        for (int s=0; s<nbsearch; s++)
        {
            int idx = seismic_tinyname[i].indexOf(list1[s], 0, Qt::CaseInsensitive);
            if ( idx >=0 || prefix.isEmpty() ) val++;
        }
        if ( val == nbsearch || nbsearch == 0  )
        {
        	QListWidgetItem *item = new QListWidgetItem;
        	item->setText(seismic_tinyname[i]);
        	item->setToolTip(seismic_tinyname[i]);
        	item->setForeground(this->seismic_diplay_color[i]);
        	this->lw_seismic->addItem(item);
            // this->lw_seismic->addItem(seismic_tinyname[i]);
            // if ( this->seismic_diplay_color.size() != 0 )
            // {
            //     this->lw_seismic->item(cpt)->setForeground(this->seismic_diplay_color[i]);
            // }
            // cpt++;
        }            
    }
}

void GeotimeProjectManagerWidget::display_nurbs_list(QString prefix)
{
    this->lw_nurbs->clear();
    QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
    int nbsearch = list1.size();

    // int cpt = 0;
    for (int i=0; i<this->nurbs_tinyname.size(); i++)
    {
         // int idx = seismic_tinyname[i].indexOf(prefix, 0, Qt::CaseInsensitive);

         int val = 0;
        for (int s=0; s<nbsearch; s++)
        {
            int idx = nurbs_tinyname[i].indexOf(list1[s], 0, Qt::CaseInsensitive);
            if ( idx >=0 || prefix.isEmpty() ) val++;
        }
        if ( val == nbsearch || nbsearch == 0  )
        {
        	QListWidgetItem *item = new QListWidgetItem;
        	item->setText(nurbs_tinyname[i]);
        	item->setToolTip(nurbs_tinyname[i]);

        	this->lw_nurbs->addItem(item);
            // this->lw_seismic->addItem(seismic_tinyname[i]);
            // if ( this->seismic_diplay_color.size() != 0 )
            // {
            //     this->lw_seismic->item(cpt)->setForeground(this->seismic_diplay_color[i]);
            // }
            // cpt++;
        }
    }
}


void GeotimeProjectManagerWidget::display_seismic_basket_list()
{   
    this->lw_seismic_basket->clear();
    int cpt = 0;
    for (int i=0; i<this->seismic_tinyname_basket.size(); i++)
    {
    	QListWidgetItem *item = new QListWidgetItem;
    	item->setText(this->seismic_tinyname_basket[i]);
    	item->setToolTip(this->seismic_tinyname_basket[i]);
    	if ( this->seismic_basket_color.size() > i ) item->setForeground(this->seismic_basket_color[i]);
    	this->lw_seismic_basket->addItem(item);
        // this->lw_seismic_basket->addItem(this->seismic_tinyname_basket[i]);
    }
}

void GeotimeProjectManagerWidget::display_nurbs_basket_list()
{
    this->lw_nurbs_basket->clear();
    int cpt = 0;
    for (int i=0; i<this->nurbs_tinyname_basket.size(); i++)
    {
    	QListWidgetItem *item = new QListWidgetItem;
    	item->setText(this->nurbs_tinyname_basket[i]);
    	item->setToolTip(this->nurbs_tinyname_basket[i]);
      	this->lw_nurbs_basket->addItem(item);
        // this->lw_seismic_basket->addItem(this->seismic_tinyname_basket[i]);
    }
}

QString GeotimeProjectManagerWidget::getWellsHeadBasketSelectedName()
{
	QList<QListWidgetItem*> list0 = lw_wellsbasket->selectedItems();
	if ( list0.empty() ) return "";
	return list0[0]->text();
}



int GeotimeProjectManagerWidget::well_wellbore_display_valid(QString prefix, int i_well, int i_bore)
{
	if ( prefix.compare("") == 0 ) return 1;
	std::vector<std::vector<QString>> list_prefix = multiCriterionSearchFormat(prefix);
	int ret0 = 0, ret1 = 0, ret2 = 0, ret3 = 0;

	int ret = qstring_display_listPrefix_valid(display0.wells[i_well].bore[i_bore].bore_tinyname, list_prefix[0]);
	if ( ret == 1 )
		ret0 = 1;

	ret1 = 0;
	if ( display0.wells[i_well].bore[i_bore].log_tinyname.size() == 0 && list_prefix[1].size() == 0 )
		ret1 = 1;
	else
		for (int i=0; i<display0.wells[i_well].bore[i_bore].log_tinyname.size(); i++ )
		{
			ret = qstring_display_listPrefix_valid(display0.wells[i_well].bore[i_bore].log_tinyname[i], list_prefix[1]);
			if ( ret == 1 )
				ret1 = 1;
		}

	ret2 = 0;
	if ( display0.wells[i_well].bore[i_bore].tf2p_tinyname.size() == 0 && list_prefix[2].size() == 0 )
		ret2 = 1;
	else
		for (int i=0; i<display0.wells[i_well].bore[i_bore].tf2p_tinyname.size(); i++ )
		{
			ret = qstring_display_listPrefix_valid(display0.wells[i_well].bore[i_bore].tf2p_tinyname[i], list_prefix[2]);
			if ( ret == 1 ) ret2 = 1;
		}

	/*
	ret3 = 0;
	// if ( display0.wells[i_well].bore[i_bore].picks_tinyname.size() == 0 ) ret3 = 1;
	for (int i=0; i<display0.wells[i_well].bore[i_bore].picks_tinyname.size(); i++ )
	{
		ret = qstring_display_listPrefix_valid(display0.wells[i_well].bore[i_bore].picks_tinyname[i], list_prefix[3]);
		if ( ret == 1 ) ret3 = 1;
	}
	if ( display0.wells[i_well].bore[i_bore].picks_tinyname.size() == 0 && list_prefix[1].size() != 0 ) ret1 = 0;
	*/
	int res = ret0*ret1*ret2;
	if ( res == 0 ) return 0;
	return 1;
	// return ret0*ret1*ret2*ret3;
}


void GeotimeProjectManagerWidget::display_wells_list(QString prefix)
{
	qDebug() << prefix;
	this->lw_wells->clear();

	for (int i_well=0; i_well<display0.wells.size(); i_well++)
	{
		for (int i_bore=0; i_bore<display0.wells[i_well].bore.size(); i_bore++)
		{

			if ( well_wellbore_display_valid(prefix, i_well, i_bore) )
			{
				QString name = display0.wells[i_well].bore[i_bore].bore_tinyname;
				QListWidgetItem *item = new QListWidgetItem;
				item->setText(name);
				item->setToolTip(name);
				item->setData(Qt::UserRole, display0.wells[i_well].bore[i_bore].bore_fullname);
				this->lw_wells->addItem(item);

			}
		}
	}
}

void GeotimeProjectManagerWidget::display_horizons_list(QString prefix)
{
    this->lw_horizons->clear();
    QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
    int nbsearch = list1.size(); 

    for (int i=0; i<this->horizons_tinyname.size(); i++)
    {
        // int idx = horizons_tinyname[i].indexOf(prefix, 0, Qt::CaseInsensitive);
        int val = 0;
        for (int s=0; s<nbsearch; s++)
        {
            int idx = horizons_tinyname[i].indexOf(list1[s], 0, Qt::CaseInsensitive);
            if ( idx >=0 || prefix.isEmpty() ) val++;
        }
        if ( val == nbsearch || nbsearch == 0 )
        {
        	QListWidgetItem *item = new QListWidgetItem;
        	item->setText(horizons_tinyname[i]);
        	item->setToolTip(horizons_tinyname[i]);
        	this->lw_horizons->addItem(item);
        }
    }
}

void GeotimeProjectManagerWidget::display_horizons_basket_list()
{   
    this->lw_horizons_basket->clear();
    int cpt = 0;
    for (int i=0; i<this->horizons_tinyname_basket.size(); i++)
    {
    	QListWidgetItem *item = new QListWidgetItem;
    	item->setText(this->horizons_tinyname_basket[i]);
    	item->setToolTip(this->horizons_tinyname_basket[i]);
    	this->lw_horizons_basket->addItem(item);
        // this->lw_horizons_basket->addItem(this->horizons_tinyname_basket[i]);
    }
}

void GeotimeProjectManagerWidget::display_cultural_list(QString prefix)
{
	this->lw_cultural->clear();
	for (int n=0; n<display0.culturals.cdata_tinyname.size(); n++)
	{
		if ( qstring_display_valid(display0.culturals.cdata_tinyname[n], prefix) )
		{
			QString txt = display0.culturals.cdata_tinyname[n];
			QListWidgetItem *item = new QListWidgetItem;
			item->setText(QString(txt));
			item->setToolTip(QString(txt));
			item->setForeground(Qt::yellow);
			this->lw_cultural->addItem(item);
		}
	}
	for (int n=0; n<display0.culturals.strd_tinyname.size(); n++)
	{
		if ( qstring_display_valid(display0.culturals.strd_tinyname[n], prefix) )
		{
			QString txt = display0.culturals.strd_tinyname[n];
			QListWidgetItem *item = new QListWidgetItem;
			item->setText(QString(txt));
			item->setToolTip(QString(txt));
			item->setForeground(Qt::green);
			this->lw_cultural->addItem(item);
		}
	}
}


void GeotimeProjectManagerWidget::display_culturals_basket_list()
{
	this->lw_cultural_basket->clear();
	for (int n=0; n<data0.culturals.cdata_tinyname.size(); n++)
	{
		QString txt = data0.culturals.cdata_tinyname[n];
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(QString(txt));
		item->setToolTip(QString(txt));
		item->setForeground(Qt::yellow);
		this->lw_cultural_basket->addItem(item);
	}
	for (int n=0; n<data0.culturals.strd_tinyname.size(); n++)
	{
		QString txt = data0.culturals.strd_tinyname[n];
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(QString(txt));
		item->setToolTip(QString(txt));
		item->setForeground(Qt::green);
		this->lw_cultural_basket->addItem(item);
	}
}


void GeotimeProjectManagerWidget::display_neurons_list(QString prefix)
{
    this->lw_neurons->clear();
    QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
    int nbsearch = list1.size();       
    for (int i=0; i<this->neurons_tinyname.size(); i++)
    {
         // int idx = neurons_tinyname[i].indexOf(prefix, 0, Qt::CaseInsensitive);
        int val = 0;
        for (int s=0; s<nbsearch; s++)
        {
            int idx = neurons_tinyname[i].indexOf(list1[s], 0, Qt::CaseInsensitive);
            if ( idx >=0 || prefix.isEmpty() ) val++;
        }         
        if ( val == nbsearch || nbsearch == 0 )
        {
        	QListWidgetItem *item = new QListWidgetItem;
        	item->setText(neurons_tinyname[i]);
        	item->setToolTip(neurons_tinyname[i]);
        	this->lw_neurons->addItem(item);
         //  this->lw_neurons->addItem(neurons_tinyname[i]);
        }
    }
}

QFileInfoList GeotimeProjectManagerWidget::get_dirlist(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();
    return list;    
}

/*
std::vector<char*> GeotimeProjectManagerWidget::get_dirlist2(QString path)
{
    std::vector<char*> tab;
    FILE *pfile;
    
    char buff[10000];
    fprintf(stderr, "path: %s\n",  path.toStdString().c_str());
    sprintf(buff, "find %s -mindepth 1 -maxdepth 1 -exec basename \\{} \\; | sort", path.toStdString().c_str());
    if ( ( pfile = popen(buff, "r") ) == NULL )
    {
        fprintf (stderr, "erreur");
        return tab;
    }   
    while (fgets (buff, sizeof(buff), pfile) != NULL)
    {
        strtok(buff, "\n");
        char *b = (char*)malloc(strlen(buff)+1);
        strcpy(b, buff);
        if ( buff[0] != 0 && strcmp(buff, ".") != 0 ) tab.push_back(b);
    }  
    fclose(pfile);
    return tab;
}
*/

std::vector<QString> GeotimeProjectManagerWidget::get_dirlist2(QString path)
{
    std::vector<QString> tab;
    FILE *pfile;

    char buff[10000];
    fprintf(stderr, "path: %s\n",  path.toStdString().c_str());
    sprintf(buff, "find %s -mindepth 1 -maxdepth 1 -exec basename \\{} \\; | sort", path.toStdString().c_str());
    if ( ( pfile = popen(buff, "r") ) == NULL )
    {
        fprintf (stderr, "erreur");
        return tab;
    }
    while (fgets (buff, sizeof(buff), pfile) != NULL)
    {
        strtok(buff, "\n");
        // char *b = (char*)malloc(strlen(buff)+1);
        // strcpy(b, buff);
        QString txt = QString(buff);
        if ( txt.compare(QString("")) !=0 && txt.compare(QString("."))) tab.push_back(txt);
        // if ( buff[0] != 0 && strcmp(buff, ".") != 0 ) tab.push_back(b);
    }
    fclose(pfile);
    return tab;
}



void GeotimeProjectManagerWidget::seismic_names_database_update()
{
	QString db_filename = get_seismic_database_name();
	std::vector<SeismisDatabaseManager::Data> data0 = SeismisDatabaseManager::databaseRead(db_filename);
	int N = data0.size();
	seismic_tinyname.resize(N);
	seismic_fullname.resize(N);
	seismic_diplay_color.resize(N);
	m_seismic_fullpath.resize(N);
	for (int i=0; i<N; i++)
	{
		seismic_tinyname[i] = data0[i].name;
		seismic_fullname[i] = data0[i].fullname;
		seismic_diplay_color[i] = data0[i].color;
		m_seismic_fullpath[i] = data0[i].path;
	}

	/*
	QString db_filename = get_seismic_database_name();
	fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());

	seismic_tinyname.clear();
	seismic_fullname.clear();
	seismic_diplay_color.clear();

	int N = 0, n0 = 0;
	char buff[100000], buff2[10000], buff3[100000];
	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	fscanf(pFile, "Seismic database\n", buff);
	fscanf(pFile, "Seismic number: %d\n", &N);
	seismic_tinyname.resize(N);
	seismic_fullname.resize(N);
	seismic_diplay_color.resize(N);
	for (int n=0; n<N; n++)
	{
		int res = fscanf(pFile, "%d %[^;];%[^;];%[^\n]\n", &n0, buff, buff2, buff3);
		//qDebug()<<" ===> "<<res<< " , no "<<n0<<" , "<< buff<<" , "<< buff2<<" , "<<buff3;
		seismic_tinyname[n] = QString(buff);
		seismic_fullname[n] = QFileInfo(QString(buff2)).fileName(); // because database contains absolute paths, but object used to accept file names
		QBrush brush = Qt::white;
		if ( strcmp(buff3, SeismicManager::TIME_SHORT_COLOR_STR.toStdString().c_str()) == 0 ) brush = SeismicManager::TIME_SHORT_COLOR;
		if ( strcmp(buff3, SeismicManager::TIME_8BIT_COLOR_STR.toStdString().c_str()) == 0 ) brush = SeismicManager::TIME_8BIT_COLOR;
		if ( strcmp(buff3, SeismicManager::TIME_32BIT_COLOR_STR.toStdString().c_str()) == 0 ) brush = SeismicManager::TIME_32BIT_COLOR;
		if ( strcmp(buff3, SeismicManager::DEPTH_SHORT_COLOR_STR.toStdString().c_str()) == 0 ) brush = SeismicManager::DEPTH_SHORT_COLOR;
		if ( strcmp(buff3, SeismicManager::DEPTH_8BIT_COLOR_STR.toStdString().c_str()) == 0 ) brush = SeismicManager::DEPTH_8BIT_COLOR;
		if ( strcmp(buff3, SeismicManager::DEPTH_32BIT_COLOR_STR.toStdString().c_str()) == 0 ) brush = SeismicManager::DEPTH_32BIT_COLOR;
		if ( strcmp(buff3, "green") == 0 ) brush = Qt::green;
		seismic_diplay_color[n] = brush;
	}
	fclose(pFile);
	*/
}

void GeotimeProjectManagerWidget::seismic_names_database_create()
{
	FILE *pFile = NULL;
	QString db_filename = get_seismic_database_name();
	QString seismic_path = get_seismic_path0();
	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	int N = seismic_fullname.size();
	fprintf(pFile, "Seismic database\n");
	fprintf(pFile, "Seismic number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		QString seismic_color = "white";
		if ( seismic_diplay_color.size() > 0 )
		{
			if ( seismic_diplay_color[n] == SeismicManager::TIME_SHORT_COLOR ) seismic_color = SeismicManager::TIME_SHORT_COLOR_STR;
			else if ( seismic_diplay_color[n] == SeismicManager::TIME_8BIT_COLOR ) seismic_color = SeismicManager::TIME_8BIT_COLOR_STR;
			else if ( seismic_diplay_color[n] == SeismicManager::TIME_32BIT_COLOR ) seismic_color = SeismicManager::TIME_32BIT_COLOR_STR;
			else if ( seismic_diplay_color[n] == SeismicManager::DEPTH_SHORT_COLOR ) seismic_color = SeismicManager::DEPTH_SHORT_COLOR_STR;
			else if ( seismic_diplay_color[n] == SeismicManager::DEPTH_8BIT_COLOR ) seismic_color = SeismicManager::DEPTH_8BIT_COLOR_STR;
			else if ( seismic_diplay_color[n] == SeismicManager::DEPTH_32BIT_COLOR ) seismic_color = SeismicManager::DEPTH_32BIT_COLOR_STR;
			else if ( seismic_diplay_color[n] == Qt::green ) seismic_color = QString("green");
		}

		QString absolute_seismic_path = seismic_path + seismic_fullname[n];

		fprintf(pFile, "%d %s;%s;%s\n", n, seismic_tinyname[n].toStdString().c_str(), absolute_seismic_path.toStdString().c_str(), seismic_color.toStdString().c_str());
	}
	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}

void GeotimeProjectManagerWidget::seismic_names_disk_update()
{
    char buff[1000];

    QString path = get_seismic_path0();
    
    // QFileInfoList list = get_seismic_list(path);
    std::vector<QString> list = get_seismic_list2(path);

    int N = list.size();
    seismic_fullname.resize(N);
	seismic_tinyname.resize(N);    
    seismic_diplay_color.resize(N);
    int newListIndex = 0;
    for (int i=0; i<list.size(); i++)
    {
        // QFileInfo fileInfo = list.at(i);
        // QString filename = fileInfo.fileName();
        bool isValid = true;

        QString filename = list[i];

        // qDebug() << "Try add seismic file : " << filename;
        seismic_fullname[newListIndex] = filename;
        int lastPoint = filename.lastIndexOf(".");
        QString fileNameNoExt = filename.left(lastPoint);        
        QString ext = filename.right(filename.size()-lastPoint-1);
        QString desc_filename = path + fileNameNoExt + ".desc";
        FILE *pfile = fopen(desc_filename.toStdString().c_str(), "r");
        int ok = 0;
        if ( pfile != NULL && ext.compare("xt") == 0 )
        {
            fgets(buff, 10000, pfile);
            fgets(buff, 10000, pfile);
            fgets(buff, 10000, pfile);
            buff[0] = 0; fscanf(pfile, "name=%s\n", buff);
            fclose(pfile);
            QString tmp = QString(buff);
            if ( !tmp.isEmpty() )
            {
                seismic_tinyname[newListIndex] = QString(tmp);
                ok = 1;
            }
        }
        if ( ok == 0 )
        {
            QString tmp = filename;
            QString header = tmp.left(10);
            tmp = tmp.left(lastPoint);
            if ( header.compare(QString("seismic3d.")) == 0 )
            {
                tmp.remove(0, 10);
            }
            // int lastPoint = tmp.lastIndexOf(".");
            // QString ext = tmp.right(tmp.size()-lastPoint-1);

            if ( ext.compare("cwt") == 0 )
            {
                tmp = tmp + QString(" (compress)");
            }
            seismic_tinyname[newListIndex] = tmp;
        }
        if ( ext.compare("xt") == 0 )
        {
        	QString tmp_filename = path + seismic_fullname[newListIndex];
        	int axis = filext_axis(tmp_filename);
        	inri::Xt::Type type = inri::Xt::Unknown;
			if (axis==0 || axis==1) {
				inri::Xt xt(tmp_filename.toStdString().c_str());
				if (xt.is_valid()) {
					type = xt.type();
				}
			}
			if (axis == 0 ) {
				switch (type) {
				case inri::Xt::Signed_16:
					this->seismic_diplay_color[newListIndex] = SeismicManager::TIME_SHORT_COLOR;
					break;
				case inri::Xt::Unsigned_16:
				case inri::Xt::Unsigned_8:
				case inri::Xt::Signed_8:
					this->seismic_diplay_color[newListIndex] = SeismicManager::TIME_8BIT_COLOR;
					break;
				default:
					this->seismic_diplay_color[newListIndex] = SeismicManager::TIME_32BIT_COLOR;
					break;
				}
			} else if (axis == 1 ) {
				switch (type) {
				case inri::Xt::Signed_16:
					this->seismic_diplay_color[newListIndex] = SeismicManager::DEPTH_SHORT_COLOR;
					break;
				case inri::Xt::Unsigned_16:
				case inri::Xt::Unsigned_8:
				case inri::Xt::Signed_8:
					this->seismic_diplay_color[newListIndex] = SeismicManager::DEPTH_8BIT_COLOR;
					break;
				default:
					this->seismic_diplay_color[newListIndex] = SeismicManager::DEPTH_32BIT_COLOR;
					break;
				}
			} else  {
                this->seismic_diplay_color[newListIndex] = Qt::white;
                isValid = false;
        	}
        }
        else
        {
        	this->seismic_diplay_color[newListIndex] = Qt::green;
        }
        if (isValid) {
            newListIndex++;
        }
    }
    if (newListIndex<N) {
        seismic_fullname.resize(newListIndex);
        seismic_tinyname.resize(newListIndex);
        seismic_diplay_color.resize(newListIndex);
    }
}


void GeotimeProjectManagerWidget:: nurbs_names_disk_update()
{
   char buff[1000];

    QString path = get_nurbs_path0();

    // QFileInfoList list = get_seismic_list(path);
    std::vector<QString> list = get_nurbs_list(path);

    int N = list.size();


    nurbs_fullname.resize(N);
	nurbs_tinyname.resize(N);

 //   int newListIndex = 0;


	for (int i=0; i<N; i++)
	{

		nurbs_tinyname[i] = list[i];
		nurbs_fullname[i] = path + list[i];

	/*	QFile file(nurbs_fullname[i]);
		if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
		{
			qDebug()<<" Load nurbs ouverture du fichier impossible :"<<path;
			return;
		}

		//QString alllines;

		QTextStream in(&file);

		int nbCurve = 0;

		while(!in.atEnd())
		{
			QString line = in.readLine();
			QStringList line1 = line.split("|");

			if(line1[0]=="color")
			{
				QStringList color1 =line1[1].split("-");
				QStringList color2 =line1[2].split("-");
				QColor  col1 (color1[0].toInt(),color1[1].toInt(),color1[2].toInt());
				QColor  col2 (color2[0].toInt(),color2[1].toInt(),color2[2].toInt());
				//QVector3D pos(line1[1].toFloat(),line1[2].toFloat(),line1[3].toFloat());

			}
			else if(line1[0]=="precision")
			{
				int precision = line1[1].toInt();
				int precis = 0;
				if(line1.count()>2) line1[2].toInt();
			}
			else if(line1[0]=="directrice")
			{
				int nb = line1[1].toInt();
				int val=0;
				if(line1.count()>2) val = line1[2].toInt();
				for(int indice =3;indice <nb+3;indice+=3)
				{
					//QVector3D pos(line1[indice].toFloat(),line1[indice+1].toFloat(),line1[indice+2].toFloat());
				}
			}
			else if(line1[0]=="nbcurve")
			{
				nbCurve = line1[1].toInt();
			}
			else if(line1[0]=="nbpts")
			{
				int nbpts = line1[1].toInt();
				int val = 0;
				if(line1.count()>2)  val = line1[2].toInt();
				for(int indice =3;indice <nbpts+3;indice+=3)
				{
					//QVector3D pos(line1[indice].toFloat(),line1[indice+1].toFloat(),line1[indice+2].toFloat());
				}

			}
			else if(line1[0]=="randomTransformation")
			{
				int tr1 = line1[1].toInt();
				int tr2 = line1[2].toInt();
			}
			else if(line1[0]=="poly")
			{
				int nb = line1[1].toInt();
			}
			else if(line1[0]=="affine")
			{
				int aff1 = line1[1].toInt();
				int aff2 = line1[2].toInt();
			}
			else if(line1[0]=="direct")
			{
				float d1 =line1[1].toFloat();
				float d2 =line1[2].toFloat();
				float d3 =line1[3].toFloat();
				float d4 =line1[4].toFloat();
				float d5 =line1[5].toFloat();
				float d6 =line1[6].toFloat();
			}
		}*/


	}


}


void GeotimeProjectManagerWidget::seismic_names_update()
{
	QString db_filename = get_seismic_database_name();
	if (  QFile::exists(db_filename) )
	{
		seismic_names_database_update();
	}
	else
	{
		seismic_names_disk_update();
		seismic_names_database_create();
	}
}


void GeotimeProjectManagerWidget::seismic_basket_add()
{
	QList<QListWidgetItem*> list0 = lw_seismic->selectedItems();
	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		int idx = getIndexFromVectorString(seismic_tinyname, txt);
		if ( idx >= 0 )
		{
			int idx_basket = getIndexFromVectorString(m_seismic_fullpath_basket, m_seismic_fullpath[idx]);
			if ( idx_basket < 0 )
			{
				seismic_tinyname_basket.push_back(seismic_tinyname[idx]);
				m_seismic_fullpath_basket.push_back(m_seismic_fullpath[idx]);
				seismic_basket_color.push_back(seismic_diplay_color[idx]);
			}
		}
	}
	/*
    QList<QListWidgetItem*> list0 = lw_seismic->selectedItems();   
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(seismic_tinyname, txt);
        if ( idx >= 0 )
        {
            int idx_basket = getIndexFromVectorString(seismic_fullname_basket, seismic_fullname[idx]);
            if ( idx_basket < 0 )
            {
                seismic_tinyname_basket.push_back(seismic_tinyname[idx]);
                seismic_fullname_basket.push_back(seismic_fullname[idx]);
                seismic_basket_color.push_back(seismic_diplay_color[idx]);
            }
        }
    }
    */
}


void GeotimeProjectManagerWidget::seismic_basket_sub()
{
    QList<QListWidgetItem*> list0 = lw_seismic_basket->selectedItems();
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(seismic_tinyname_basket, txt);
        if ( idx >= 0 )
        {
            seismic_tinyname_basket.erase(seismic_tinyname_basket.begin()+idx, seismic_tinyname_basket.begin()+idx+1);
            m_seismic_fullpath_basket.erase(m_seismic_fullpath_basket.begin()+idx, m_seismic_fullpath_basket.begin()+idx+1);
            if ( seismic_basket_color.size() > idx )
            	seismic_basket_color.erase(seismic_basket_color.begin()+idx, seismic_basket_color.begin()+idx+1);
        }
    }
}

void GeotimeProjectManagerWidget::nurbs_basket_add()
{
    QList<QListWidgetItem*> list0 = lw_nurbs->selectedItems();
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(nurbs_tinyname, txt);
        if ( idx >= 0 )
        {
            int idx_basket = getIndexFromVectorString(nurbs_fullname_basket, nurbs_fullname[idx]);
            if ( idx_basket < 0 )
            {
            	nurbs_tinyname_basket.push_back(nurbs_tinyname[idx]);
            	nurbs_fullname_basket.push_back(nurbs_fullname[idx]);

            }
        }
    }
}

void GeotimeProjectManagerWidget::nurbs_basket_sub()
{

    QList<QListWidgetItem*> list0 = lw_nurbs_basket->selectedItems();
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(nurbs_tinyname_basket, txt);
        if ( idx >= 0 )
        {
        	nurbs_tinyname_basket.erase(nurbs_tinyname_basket.begin()+idx, nurbs_tinyname_basket.begin()+idx+1);
        	nurbs_fullname_basket.erase(nurbs_fullname_basket.begin()+idx, nurbs_fullname_basket.begin()+idx+1);

        }
    }
}


void GeotimeProjectManagerWidget::horizon_names_database_update()
{
	QString db_filename = get_horizons_database_name();
	fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());

	horizons_tinyname.clear();
	horizons_fullname.clear();

	int N = 0, n0 = 0;
	char buff[100000], buff2[10000];
	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	fscanf(pFile, "Horizon database\n", buff);
	fscanf(pFile, "Horizon number: %d\n", &N);
	horizons_tinyname.resize(N);
	horizons_fullname.resize(N);
	for (int n=0; n<N; n++)
	{
		fscanf(pFile, "%d %[^;];%[^\n]\n", &n0, buff, buff2);
		horizons_tinyname[n] = QString(buff);
		horizons_fullname[n] = QString(buff2);
	}
	fclose(pFile);
}

void GeotimeProjectManagerWidget::horizon_names_database_create()
{
	FILE *pFile = NULL;
	QString db_filename = get_horizons_database_name();
	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	int N = horizons_tinyname.size();
	fprintf(pFile, "Horizon database\n");
	fprintf(pFile, "Horizon number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		fprintf(pFile, "%d %s;%s\n", n, horizons_tinyname[n].toStdString().c_str(), horizons_fullname[n].toStdString().c_str());
	}
	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}

void GeotimeProjectManagerWidget::horizon_names_disk_update()
{
	if ( !isTabHorizons() ) return;
	QString path = get_horizons_path0();
	QDir dir = QDir(path);
	dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	dir.setSorting(QDir::Name);
	QFileInfoList list = dir.entryInfoList();

    this->horizons_fullname.clear();
	this->horizons_tinyname.clear();

    int N = list.size();
    for (int i=0; i<N; i++)
    {
        QFileInfo fileInfo = list.at(i);
        QString path1 = path + fileInfo.fileName() + QString("/HORIZON_GRIDS/");
        fprintf(stderr, "horizons path1 --> %s\n", path1.toStdString().c_str());
        QDir dir0 = QDir(path1);
        dir0.setFilter(QDir::Files);
        // dir->setFilter(QDir::AllEntries );
        dir0.setSorting(QDir::Name);
        QStringList filters;
        filters << "*.raw";
        dir0.setNameFilters(filters);
        QFileInfoList list0 = getFiles(path1.toStdString(),filters);//dir0.entryInfoList();
        int NN = list0.size();
        for (int ii=0; ii<NN; ii++)
        {
            QFileInfo fileInfo0 = list0.at(ii);
            QString path0 = fileInfo0.fileName();
            QString tmp = path1 + path0;
            horizons_fullname.push_back(tmp);
            QString rawname = path0.split(".",Qt::SkipEmptyParts).at(0);
            horizons_tinyname.push_back(rawname + " (" + fileInfo.fileName() + ")");
        }
        // fprintf(stderr, "horizon --> %s\n", filename.toStdString().c_str());
    }
}


void GeotimeProjectManagerWidget::horizon_names_update()
{
	QString db_filename = get_horizons_database_name();
	if (  QFile::exists(db_filename) )
	{
		horizon_names_database_update();
	}
	else
	{
		horizon_names_disk_update();
		horizon_names_database_create();
	}
}








void GeotimeProjectManagerWidget::horizons_basket_add()
{
    QList<QListWidgetItem*> list0 = lw_horizons->selectedItems();   
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(horizons_tinyname, txt);
        if ( idx >= 0 )
        {
            int idx_basket = getIndexFromVectorString(horizons_fullname_basket, horizons_fullname[idx]);
            if ( idx_basket < 0 )
            {
                horizons_tinyname_basket.push_back(horizons_tinyname[idx]);
                horizons_fullname_basket.push_back(horizons_fullname[idx]);
            }
        }
    }
}

void GeotimeProjectManagerWidget::horizons_basket_sub()
{
    QList<QListWidgetItem*> list0 = lw_horizons_basket->selectedItems();
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(horizons_tinyname_basket, txt);
        if ( idx >= 0 )
        {
            horizons_tinyname_basket.erase(horizons_tinyname_basket.begin()+idx, horizons_tinyname_basket.begin()+idx+1);
            horizons_fullname_basket.erase(horizons_fullname_basket.begin()+idx, horizons_fullname_basket.begin()+idx+1);
        }
    }
}





void GeotimeProjectManagerWidget::rgb_basket_add()
{
    QList<QListWidgetItem*> list0 = qlw_rgb->selectedItems();
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(rgb_tinyname, txt);
        if ( idx >= 0 )
        {
            int idx_basket = getIndexFromVectorString(rgb_basket_fullname, rgb_fullname[idx]);
            if ( idx_basket < 0 )
            {
                rgb_basket_tinyname.push_back(rgb_tinyname[idx]);
                rgb_basket_fullname.push_back(rgb_fullname[idx]);
            }
        }
    }
}


void GeotimeProjectManagerWidget::rgb_basket_sub()
{
    QList<QListWidgetItem*> list0 = qlw_rgb_basket->selectedItems();
    for (int i=0; i<list0.size(); i++)
    {
        QString txt = list0[i]->text();
        int idx = getIndexFromVectorString(rgb_basket_tinyname, txt);
        if ( idx >= 0 )
        {
        	rgb_basket_tinyname.erase(rgb_basket_tinyname.begin()+idx, rgb_basket_tinyname.begin()+idx+1);
        	rgb_basket_fullname.erase(rgb_basket_fullname.begin()+idx, rgb_basket_fullname.begin()+idx+1);
        }
    }
}
































void GeotimeProjectManagerWidget::wells_names_disk_update()
{
	if ( !isTabWells() ) return;
	QString db_filename = get_well_database_name();
	QString path = get_wells_path0();
	WellsDatabaseManager::update(db_filename, path);
	return;
	/*

    // QString path = get_wells_path0();
    QFileInfoList list = get_dirlist(path);
    int N = list.size();

    display0.wells.resize(N);
    for (int n_well=0; n_well<N; n_well++)
    {
    	QFileInfo fileInfo = list[n_well];
    	QString filetinyname = fileInfo.fileName();
    	QString filefullname = fileInfo.absoluteFilePath();

    	QDir headDir(filefullname);
    	QString headDescName = filetinyname + ".desc";
    	if (headDir.exists(headDescName)) {
    		QString descFile = headDir.absoluteFilePath(headDescName);
    		QString name = ProjectManagerNames::getKeyTabFromFilename(descFile, "Name");
    		if (!name.isNull() && !name.isEmpty()) {
    			filetinyname = name;
    		}
    	}
    	display0.wells[n_well].head_tinyname = filetinyname;
    	display0.wells[n_well].head_fullname = filefullname;

    	QFileInfoList list_bore = get_dirlist(filefullname);
    	int Nbores = list_bore.size();
    	display0.wells[n_well].bore.resize(Nbores);

    	for (int n_bore=0; n_bore<Nbores; n_bore++)
    	{
    		QFileInfo bore_fileInfo = list_bore[n_bore];
    		QString bore_filetinyname = bore_fileInfo.fileName();
    		QString bore_filefullname = bore_fileInfo.absoluteFilePath();

    		QDir boreDir(bore_filefullname);
    		QString boreDescName = bore_filetinyname + ".desc";
    		if (boreDir.exists(boreDescName)) {
    			QString descFile = boreDir.absoluteFilePath(boreDescName);
    			QString name = ProjectManagerNames::getKeyTabFromFilename(descFile, "Name");
    			if (!name.isNull() && !name.isEmpty()) {
    				bore_filetinyname = name;
    			}
    		}
    		display0.wells[n_well].bore[n_bore].bore_tinyname = QString("[ ") + filetinyname +QString(" ] ") + bore_filetinyname;
    		display0.wells[n_well].bore[n_bore].bore_fullname = bore_filefullname;

    		welldeviation_names_update(bore_filefullname, n_well, n_bore);
    		welllog_names_update(bore_filefullname, n_well, n_bore);
    		welltf2p_names_update(bore_filefullname, n_well, n_bore);
    		wellpicks_names_update(bore_filefullname, n_well, n_bore);
    	}
    	fprintf(stderr, "%d %d\n", n_well, N);
    }
    */
}

void GeotimeProjectManagerWidget::wells_names_database_create()
{
	FILE *pFile = NULL;
	QString db_filename = get_well_database_name();
	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	fprintf(pFile, "Wells database\n");
	fprintf(pFile, "Wells number: %d\n", display0.wells.size());
	for (int n=0; n<display0.wells.size(); n++)
	{
	    fprintf(pFile, "head:%d %s;%s\n", n, display0.wells[n].head_tinyname.toStdString().c_str(), display0.wells[n].head_fullname.toStdString().c_str());
	    int Nbore = display0.wells[n].bore.size();
	    fprintf(pFile, "bore number: %d\n", Nbore);
	    for (int n2=0; n2<Nbore; n2++)
	    {
	    	fprintf(pFile, "head: %d bore: %d %s;%s\n", n, n2, display0.wells[n].bore[n2].bore_tinyname.toStdString().c_str(), display0.wells[n].bore[n2].bore_fullname.toStdString().c_str());
	    	int N2 = display0.wells[n].bore[n2].log_tinyname.size();
	    	fprintf(pFile, "logs number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d log: %d %s;%s\n", n, n2, n3, display0.wells[n].bore[n2].log_tinyname[n3].toStdString().c_str(), display0.wells[n].bore[n2].log_fullname[n3].toStdString().c_str());
	    	}
	    	N2 = display0.wells[n].bore[n2].tf2p_tinyname.size();
	    	fprintf(pFile, "tf2p number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d tf2p: %d %s;%s\n", n, n2, n3, display0.wells[n].bore[n2].tf2p_tinyname[n3].toStdString().c_str(), display0.wells[n].bore[n2].tf2p_fullname[n3].toStdString().c_str());
	    	}
	    	N2 = display0.wells[n].bore[n2].picks_tinyname.size();
	    	fprintf(pFile, "picks number: %d\n", N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fprintf(pFile, "head: %d bore: %d picks: %d %s;%s\n", n, n2, n3, display0.wells[n].bore[n2].picks_tinyname[n3].toStdString().c_str(), display0.wells[n].bore[n2].picks_fullname[n3].toStdString().c_str());
	    	}
	    }
	}
	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}


void GeotimeProjectManagerWidget::wells_names_database_update()
{
	QString db_filename = get_well_database_name();
	fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());

	if ( !isTabWells() ) return;
	QString path = get_wells_path0();

	display0.wells.clear();
	int nwells = 0;
	char buff[100000], buff2[10000];
	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	fscanf(pFile, "Wells database\n", buff);
	fscanf(pFile, "Wells number: %d\n", &nwells);
	display0.wells.resize(nwells);

	int n0, t1, t2, t3;
	int Nbore, N2;
	for (int n=0; n<nwells; n++)
	{
		fscanf(pFile, "head: %d %[^;];%[^\n]\n", &n0, buff, buff2);
		display0.wells[n].head_tinyname = QString(buff);
		display0.wells[n].head_fullname = QString(buff2);
		fscanf(pFile, "bore number: %d\n", &Nbore);
		display0.wells[n].bore.resize(Nbore);
		for (int n2=0; n2<Nbore; n2++)
		{
			fscanf(pFile, "head: %d bore:%d %[^;];%[^\n]\n", &t1, &t2, buff, buff2);
			display0.wells[n].bore[n2].bore_tinyname = QString(buff);
			display0.wells[n].bore[n2].bore_fullname = QString(buff2);
			fscanf(pFile, "logs number: %d\n", &N2);
			display0.wells[n].bore[n2].log_tinyname.resize(N2);
			display0.wells[n].bore[n2].log_fullname.resize(N2);
			for (int n3=0; n3<N2; n3++)
			{
				fscanf(pFile, "head:%d bore:%d log:%d %[^;];%[^\n]\n", &t1, &t2, &t3, buff, buff2);
				display0.wells[n].bore[n2].log_tinyname[n3] = QString(buff);
				display0.wells[n].bore[n2].log_fullname[n3] = QString(buff2);
	    	}
	   		fscanf(pFile, "tf2p number: %d\n", &N2);
	   		display0.wells[n].bore[n2].tf2p_tinyname.resize(N2);
	   		display0.wells[n].bore[n2].tf2p_fullname.resize(N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fscanf(pFile, "head:%d bore:%d tf2p:%d %[^;];%[^\n]\n", &t1, &t2, &t3, buff, buff2);
	    		display0.wells[n].bore[n2].tf2p_tinyname[n3] = QString(buff);
	    		display0.wells[n].bore[n2].tf2p_fullname[n3] = QString(buff2);
	    	}
	    	fscanf(pFile, "picks number: %d\n", &N2);
	    	display0.wells[n].bore[n2].picks_tinyname.resize(N2);
	    	display0.wells[n].bore[n2].picks_fullname.resize(N2);
	    	for (int n3=0; n3<N2; n3++)
	    	{
	    		fscanf(pFile, "head:%d bore:%d picks:%d %[^;];%[^\n]\n", &t1, &t2, &t3, buff, buff2);
	    		display0.wells[n].bore[n2].picks_tinyname[n3] = QString(buff);
	    		display0.wells[n].bore[n2].picks_fullname[n3] = QString(buff2);
	    	}
	    }
	}
	fclose(pFile);
}


void GeotimeProjectManagerWidget::wells_names_update()
{
	QString db_filename = get_well_database_name();
	if (  QFile::exists(db_filename) )
	{
		wells_names_database_update();
	}
	else
	{
		QString path = get_wells_path0();
		WellsDatabaseManager::update(db_filename, path);
		wells_names_database_update();
		// wells_names_disk_update();
		// wells_names_database_create();
	}
}

QFileInfoList GeotimeProjectManagerWidget::get_cultural_cdata_list(QString path)
{
	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.cdat";
	dir.setNameFilters(filters);
	QFileInfoList list = getFiles(path.toStdString(),filters);//dir.entryInfoList();
	return list;
}

QFileInfoList GeotimeProjectManagerWidget::get_cultural_strd_list(QString path)
{
	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.strd";
	dir.setNameFilters(filters);
	QFileInfoList list = getFiles(path.toStdString(),filters);//dir.entryInfoList();
	return list;
}


void GeotimeProjectManagerWidget::cultural_names_database_create()
{
	FILE *pFile = NULL;
	QString db_filename = get_cultural_database_name();
	fprintf(stderr, "database filename: %s\n", db_filename.toStdString().c_str());
	pFile = fopen(db_filename.toStdString().c_str(), "w");
	if ( pFile == NULL ) return;
	fprintf(pFile, "Cultural database\n");
	int N = display0.culturals.cdata_tinyname.size();
	fprintf(pFile, "cdata number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		fprintf(pFile, "%d %s;%s\n", n, display0.culturals.cdata_tinyname[n].toStdString().c_str(), display0.culturals.cdata_fullname[n].toStdString().c_str());
   	}
	N = display0.culturals.strd_tinyname.size();
	fprintf(pFile, "strd number: %d\n", N);
	for (int n=0; n<N; n++)
	{
		fprintf(pFile, "%d %s;%s\n", n, display0.culturals.strd_tinyname[n].toStdString().c_str(), display0.culturals.strd_fullname[n].toStdString().c_str());
	}
	fclose(pFile);
	chmod(db_filename.toStdString().c_str(), (mode_t)0777);
}

void GeotimeProjectManagerWidget::cultural_names_database_update()
{
	QString db_filename = get_cultural_database_name();
	fprintf(stderr, "--> %s\n", db_filename.toStdString().c_str());

	if ( !isTabCulturals() ) return;
	QString path = get_wells_path0();

	display0.culturals.cdata_tinyname.clear();
	display0.culturals.cdata_fullname.clear();
	display0.culturals.strd_tinyname.clear();
	display0.culturals.strd_fullname.clear();

	int N = 0;
	char buff[100000], buff2[10000];
	int n0, t1, t2, t3;

	FILE *pFile = NULL;
	pFile = fopen(db_filename.toStdString().c_str(), "rb");
	if ( pFile == NULL ) return;

	fscanf(pFile, "Cultural database\n", buff);
	fscanf(pFile, "cdata number: %d\n", &N);
	display0.culturals.cdata_tinyname.resize(N);
	display0.culturals.cdata_fullname.resize(N);
	for (int n=0; n<N; n++)
	{
		fscanf(pFile, "%d %[^;];%[^\n]\n", &n0, buff, buff2);
		display0.culturals.cdata_tinyname[n] = QString(buff);
		display0.culturals.cdata_fullname[n] = QString(buff2);
	}
	fscanf(pFile, "strd number: %d\n", &N);
	display0.culturals.strd_tinyname.resize(N);
	display0.culturals.strd_fullname.resize(N);
	for (int n=0; n<N; n++)
	{
		fscanf(pFile, "%d %[^;];%[^\n]\n", &n0, buff, buff2);
		display0.culturals.strd_tinyname[n] = QString(buff);
		display0.culturals.strd_fullname[n] = QString(buff2);
	}
	fclose(pFile);
}

void GeotimeProjectManagerWidget::cultural_names_disk_update()
{
	if ( !isTabCulturals() ) return;

	QString path = get_cultural_path0();
	QFileInfoList cdata_list = get_cultural_cdata_list(path);
	QFileInfoList strd_list = get_cultural_strd_list(path);
	int N = cdata_list.size();
	display0.culturals.cdata_tinyname.resize(N);
	display0.culturals.cdata_fullname.resize(N);
	for (int n=0; n<N; n++)
	{
		QFileInfo fileInfo = cdata_list.at(n);
	    QString filename = fileInfo.fileName();
	    QString fullname = path + filename;
	    display0.culturals.cdata_fullname[n] = fullname;
	    QString tinyname = fileInfo.completeBaseName();
	    FILE *pfile = fopen(fullname.toStdString().c_str(), "r");
	    if( pfile )
	    {
	    	char buff[10000];
	        fscanf(pfile, "%s\n", buff);
	        tinyname = QString(buff);
	        fclose(pfile);
	    }
	    display0.culturals.cdata_tinyname[n] = tinyname;
	}

	N = strd_list.size();
	display0.culturals.strd_tinyname.resize(N);
	display0.culturals.strd_fullname.resize(N);
	for (int n=0; n<N; n++)
	{
		QFileInfo fileInfo = strd_list.at(n);
	    QString filename = fileInfo.fileName();
	    QString fullname = path + filename;
	    display0.culturals.strd_fullname[n] = fullname;
	    QString tinyname = fileInfo.completeBaseName();
	    FILE *pfile = fopen(fullname.toStdString().c_str(), "r");
	    if( pfile )
	    {
	    	char buff[10000];
	        fscanf(pfile, "%s\n", buff);
	        tinyname = QString(buff);
	        fclose(pfile);
	    }
	    display0.culturals.strd_tinyname[n] = tinyname;
	}
}






void GeotimeProjectManagerWidget::cultural_names_update()
{
	QString db_filename = get_cultural_database_name();
	if (  QFile::exists(db_filename) )
	{
		cultural_names_database_update();
	}
	else
	{
		cultural_names_disk_update();
		cultural_names_database_create();
	}
}

void GeotimeProjectManagerWidget::trt_culturals_basket_add()
{
	QList<QListWidgetItem*> list0 = lw_cultural->selectedItems();
	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		QBrush brush = list0[i]->foreground();
		if ( brush == Qt::yellow )
		{
			int idx = getIndexFromVectorString(display0.culturals.cdata_tinyname, txt);
			if ( idx >= 0 )
			{
				int idx_basket = getIndexFromVectorString(data0.culturals.cdata_fullname, display0.culturals.cdata_fullname[idx]);
				if ( idx_basket < 0 )
				{
					data0.culturals.cdata_tinyname.push_back(display0.culturals.cdata_tinyname[idx]);
					data0.culturals.cdata_fullname.push_back(display0.culturals.cdata_fullname[idx]);
				}
			}
		}
		else
		{
			int idx = getIndexFromVectorString(display0.culturals.strd_tinyname, txt);
			int idx_basket = getIndexFromVectorString(data0.culturals.strd_tinyname, txt);
			if ( idx >= 0 )
			{
				int idx_basket = getIndexFromVectorString(data0.culturals.strd_fullname, display0.culturals.strd_fullname[idx]);
				if ( idx_basket < 0 ) {
					data0.culturals.strd_tinyname.push_back(display0.culturals.strd_tinyname[idx]);
					data0.culturals.strd_fullname.push_back(display0.culturals.strd_fullname[idx]);
				}
			}
		}
	}
	lw_cultural->clearSelection();
    display_culturals_basket_list();
}


void GeotimeProjectManagerWidget::trt_culturals_basket_sub()
{

	QList<QListWidgetItem*> list0 = lw_cultural_basket->selectedItems();
	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		QBrush brush = list0[i]->foreground();
		if ( brush == Qt::yellow )
		{
			int idx = getIndexFromVectorString(data0.culturals.cdata_tinyname, txt);
			if ( idx >= 0 )
			{
				data0.culturals.cdata_tinyname.erase(data0.culturals.cdata_tinyname.begin()+idx, data0.culturals.cdata_tinyname.begin()+idx+1);
				data0.culturals.cdata_fullname.erase(data0.culturals.cdata_fullname.begin()+idx, data0.culturals.cdata_fullname.begin()+idx+1);
			}
		}
		else
		{
			int idx = getIndexFromVectorString(data0.culturals.strd_tinyname, txt);
			if ( idx >= 0 )
			{
				data0.culturals.strd_tinyname.erase(data0.culturals.strd_tinyname.begin()+idx, data0.culturals.strd_tinyname.begin()+idx+1);
				data0.culturals.strd_fullname.erase(data0.culturals.strd_fullname.begin()+idx, data0.culturals.strd_fullname.begin()+idx+1);
			}
		}
	}
	lw_cultural_basket->clearSelection();
	display_culturals_basket_list();
}


void GeotimeProjectManagerWidget::neurons_names_update()
{
	if ( !isTabNeurons() ) return;
    QString path = get_neurons_path0();
    QFileInfoList list = get_neurons_list(path);
    int N = list.size();
    neurons_fullname.resize(N);
    neurons_tinyname.resize(N);
    for (int i=0; i<N; i++)
    {
        QFileInfo fileInfo = list.at(i);
        QString filename = fileInfo.fileName();
        QString filename_withoutext = list.at(i).completeBaseName();
        neurons_fullname[i] = filename;
        neurons_tinyname[i] = filename_withoutext;
    }
}



void GeotimeProjectManagerWidget::rgb_names_update()
{
	rgb_names_disk_update();
}


void GeotimeProjectManagerWidget::display_rgb_list(QString prefix)
{
	this->qlw_rgb->clear();
	QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
	int nbsearch = list1.size();
    for (int i=0; i<this->rgb_tinyname.size(); i++)
    {
         int val = 0;
        for (int s=0; s<nbsearch; s++)
        {
            int idx = rgb_tinyname[i].indexOf(list1[s], 0, Qt::CaseInsensitive);
	        if ( idx >=0 || prefix.isEmpty() ) val++;
	    }
	    if ( val == nbsearch || nbsearch == 0  )
	    {
	       	QListWidgetItem *item = new QListWidgetItem;
	       	item->setText(rgb_tinyname[i]);
	       	item->setToolTip(rgb_tinyname[i]);
	        this->qlw_rgb->addItem(item);
	        }
	    }
}

void GeotimeProjectManagerWidget::display_rgb_basket_list()
{
    this->qlw_rgb_basket->clear();
    int cpt = 0;
    for (int i=0; i<this->rgb_basket_tinyname.size(); i++)
    {
    	QListWidgetItem *item = new QListWidgetItem;
    	item->setText(this->rgb_basket_tinyname[i]);
    	item->setToolTip(this->rgb_basket_tinyname[i]);
    	// if ( this->seismic_basket_color.size() > i ) item->setForeground(this->seismic_basket_color[i]);
    	this->qlw_rgb_basket->addItem(item);
        // this->lw_seismic_basket->addItem(this->seismic_tinyname_basket[i]);
    }
}

void GeotimeProjectManagerWidget::rgb_names_disk_update()
{
	/*
    char buff[1000];

    QString path = get_cubeRGT2RGBPath();
    std::vector<QString> list = get_rgb_list(path);

    int N = list.size();
    rgb_fullname.resize(N);
	rgb_tinyname.resize(N);
    for (int i=0; i<list.size(); i++)
    {
        QString filename = list[i];
        QFileInfo info(filename);
        rgb_tinyname[i] = info.fileName();
        rgb_fullname[i] = path + rgb_tinyname[i];
    }
    */


   	// if ( !isTabHorizons() ) return;
   	QString path = get_horizons_path0();
    QDir dir(path);
    dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();

    this->rgb_fullname.clear();
    this->rgb_tinyname.clear();

        int N = list.size();
        for (int i=0; i<N; i++)
        {
            QFileInfo fileInfo = list.at(i);
            QString path1 = path + fileInfo.fileName() + QString("/cubeRgt2RGB/");
            QDir dir0(path1);
            dir0.setFilter(QDir::Files);
            // dir->setFilter(QDir::AllEntries );
            dir0.setSorting(QDir::Name);
            QStringList filters;
            filters << "*.raw" << "*.xt" << "*.rgb" << "*.avi";
            dir0.setNameFilters(filters);
            QFileInfoList list0 = getFiles(path1.toStdString(),filters);//dir0.entryInfoList();
            int NN = list0.size();
            for (int ii=0; ii<NN; ii++)
            {
                QFileInfo fileInfo0 = list0.at(ii);
                QString path0 = fileInfo0.fileName();
                QString tmp = path1 + path0;
                rgb_fullname.push_back(tmp);
                QString rawname = path0.split(".",Qt::SkipEmptyParts).at(0);
                QString tmpSuffix = QFileInfo(tmp).suffix().toLower();
                if (tmpSuffix.compare("rgb")==0) {
                	rawname += " (rgb)";
                } else if (tmpSuffix.compare("avi")==0) {
                	rawname += " (avi)";
                }
                rgb_tinyname.push_back(rawname);
            }
        }
}


// =============================================================


void GeotimeProjectManagerWidget::trt_projectsearchchange(QString str)
{
    // QStringList list1 = str.split(" ", Qt::SkipEmptyParts);
 
    // for (int i=0; i<list1.size(); i++)
    //     fprintf(stderr, "----> %d %s\n", i, list1[i].toStdString().c_str());
    int idx = cb_projecttype->currentIndex();
    display_project_list(idx, str);
    display_label_titles();
}


void GeotimeProjectManagerWidget::trt_surveysearchchange(QString str)
{
    display_survey_list(str);
    display_label_titles();
}

void GeotimeProjectManagerWidget::trt_surveylistClick(QListWidgetItem* p)
{
	clear_survey_specific_lists();

    seismic_names_update();
    display_seismic_list(lineedit_seismicsearch->text());
    // display_seismic_list(idx, project_name, txt, prefix);   
    horizon_names_update();
    display_horizons_list(lineedit_horizonssearch->text());

    rgb_names_update();
    display_rgb_list(lineedit_rgt2rgtsearch->text());

    display_label_titles();

    nurbs_names_disk_update();
    display_nurbs_list(lineedit_nurbssearch->text());

    // fprintf(stderr, "debug -----> %d %s\n", idx_survey, txt.toStdString().c_str());
}

void GeotimeProjectManagerWidget::trt_wellssearchchange(QString str)
{
    int idx = cb_projecttype->currentIndex();
    QListWidgetItem *p_project = lw_projetlist->currentItem();
    if ( p_project == NULL ) return;
    QString project_name = p_project->text();    
  	display_wells_list(str);
}

void GeotimeProjectManagerWidget::trt_horizonssearchchange(QString str)
{
    display_horizons_list(str);
}

void GeotimeProjectManagerWidget::trt_seismicsearchchange(QString str)
{
    display_seismic_list(str);   
    // display_seismic_list(idx, project_name, survey_name, str);
}

void GeotimeProjectManagerWidget::trt_culturalssearchchange(QString str)
{
   	display_cultural_list(str);
}

void GeotimeProjectManagerWidget::trt_neuronssearchchange(QString str)
{
   	display_neurons_list(str);
}

void GeotimeProjectManagerWidget::trt_rgbsearchchange(QString str)
{
	display_rgb_list(str);
}

void GeotimeProjectManagerWidget::trt_chkbx1(int val)
{
    bool val0 =  this->chkbx_culturals->isChecked(); this->tabw_table1->setTabEnabled(1, val0);
    val0 =  this->chkbx_wells->isChecked(); this->tabw_table1->setTabEnabled(2, val0);
    val0 =  this->chkbx_neurons->isChecked(); this->tabw_table1->setTabEnabled(3, val0);
    val0 =  this->chkbx_horizons->isChecked(); this->tabw_table1->setTabEnabled(4, val0);
}



// well_head_fullname.push_back(wells_maindir_fullname[n]);
// well_head_tinyname.push_back(wells_maindir_tinyname[n]);
// well_bore_tinyname.push_back(filename);
// well_bore_fullname.push_back(wells_maindir_fullname[n] + "/" + filename);


void GeotimeProjectManagerWidget::trt_wellbasketlistselectionchanged()
{
	QList<QListWidgetItem*> selection = lw_wellsbasket->selectedItems();
	if ( selection.size()==1 )
	{
		QString well_tiny_name = selection.first()->text();
		trt_wellbasketlistclick(well_tiny_name);
	}
	else
	{
		qlw_welllog->clear();
		qlw_welltf2p->clear();
		qlw_wellpicks->clear();
		qlw_welllog_basket->clear();
		qlw_welltf2p_basket->clear();
		qlw_wellpicks_basket->clear();
	}
}

void GeotimeProjectManagerWidget::trt_wellbasketlistclick(QListWidgetItem *p)
{

	// QString well_tiny_name = this->lw_wellsbasket->currentItem()->text();
	QString well_tiny_name = p->text();
	trt_wellbasketlistclick(well_tiny_name);
}

void GeotimeProjectManagerWidget::trt_wellbasketlistclick(QString well_tiny_name)
{
	int idx = getIndexFromVectorString(this->well_wellbore_basket, well_tiny_name);
	if ( idx == -1 ) return;
	QString bore_fullname = well_bore_fullname_basket[idx];

	// qDebug() << bore_fullname;

	welllog_names_update0(bore_fullname);
	welltf2p_names_update0(bore_fullname);
	wellpicks_names_update0(bore_fullname);
	welldeviation_names_update0(bore_fullname);

	display_welllog("", linedit_welllogsearch->text());
	display_welltf2p("", linedit_welltf2psearch->text());
	display_wellpicks("", linedit_wellpickssearch->text());

	display_welllog_basket_list();
	display_welltf2p_basket_list();
	display_wellpicks_basket_list();
}

void GeotimeProjectManagerWidget::trt_well_basket_add()
{
	QList<QListWidgetItem*> list0 = lw_wells->selectedItems();
	int idx_well, idx_bore;

	for (int i=0; i<list0.size(); i++)
	{
		QString dataPath = list0[i]->data(Qt::UserRole).toString();
		getIndexFromWellWellboreFull(dataPath, &idx_well, &idx_bore);
	    if ( idx_well >= 0 && idx_bore >= 0 )
	    {
	    	int idx_basket = getIndexFromVectorString(well_bore_fullname_basket, display0.wells[idx_well].bore[idx_bore].bore_fullname);
	    	if ( idx_basket < 0 )
	    	{
				well_wellbore_basket.push_back(display0.wells[idx_well].bore[idx_bore].bore_tinyname);
				well_head_tinyname_basket.push_back(display0.wells[idx_well].head_tinyname);
				well_head_fullname_basket.push_back(display0.wells[idx_well].head_fullname);
				well_bore_tinyname_basket.push_back(display0.wells[idx_well].bore[idx_bore].bore_tinyname);
				well_bore_fullname_basket.push_back(display0.wells[idx_well].bore[idx_bore].bore_fullname);

				QString head_tinyname = well_head_tinyname_basket.at(well_head_tinyname_basket.size()-1);
				QString head_fullname = well_head_fullname_basket.at(well_head_tinyname_basket.size()-1);
				QString bore_tinyname = well_bore_tinyname_basket.at(well_head_tinyname_basket.size()-1);
				QString bore_fullname = well_bore_fullname_basket.at(well_head_tinyname_basket.size()-1);
				QString deviationfullname = get_deviation_fullname(bore_fullname);

				int tmp_idx_well = -1, tmp_idx_bore = -1;
				welllist_create_get_index(head_tinyname, head_fullname, bore_tinyname, bore_fullname, deviationfullname, &tmp_idx_well, &tmp_idx_bore);
				if ( tmp_idx_well >= 0 && tmp_idx_bore >= 0 ) {
					// select default tfp
					QDir wellBoreDir(bore_fullname);
					QStringList descFiles = wellBoreDir.entryList(QStringList() << "*.desc", QDir::Files);
					QString wellBoreDescFile;
					QString descWellBoreFile;
					QString tfpWellBoreFile;
					if (descFiles.size()>0) {
						wellBoreDescFile = descFiles[0];
						descWellBoreFile = wellBoreDir.absoluteFilePath(wellBoreDescFile);
						tfpWellBoreFile = WellBore::getTfpFileFromDescFile(descWellBoreFile);
					}

					if (!tfpWellBoreFile.isNull() && !tfpWellBoreFile.isEmpty() && QFileInfo(tfpWellBoreFile).exists()) {
						QString tfpWellBoreName = ProjectManagerNames::getKeyTabFromFilename(tfpWellBoreFile, "Name");
						well_list[tmp_idx_well].wellborelist[tmp_idx_bore].tf2p_tinyname.clear();
						well_list[tmp_idx_well].wellborelist[tmp_idx_bore].tf2p_fullname.clear();
						well_list[tmp_idx_well].wellborelist[tmp_idx_bore].tf2p_tinyname.push_back(tfpWellBoreName);
						well_list[tmp_idx_well].wellborelist[tmp_idx_bore].tf2p_fullname.push_back(tfpWellBoreFile);
						well_list[tmp_idx_well].wellborelist[tmp_idx_bore].tf2p_displayname.push_back(tfpWellBoreName);
					}
				}
	    	}
	    }
	 }
	display_well_basket_list();
	this->lw_wells->clearSelection();
	qlw_welllog->clear();
	qlw_welltf2p->clear();
	qlw_wellpicks->clear();
}

void GeotimeProjectManagerWidget::trt_well_basket_sub()
{
	QList<QListWidgetItem*> list0 = lw_wellsbasket->selectedItems();
	if ( list0.size() <= 0 ) return;


	for (int i=0; i<list0.size(); i++)
	{
		int idx_well = -1, idx_bore = -1;
		QString txt = list0[i]->text();
		int idx = getIndexFromVectorString(well_wellbore_basket, txt);

		// fprintf(stderr, "[**] %d - %s\n", idx, txt.toStdString().c_str());

		QString name = well_bore_fullname_basket[idx];
		welllist_get_index_from_borename(name, &idx_well, &idx_bore);

		if ( idx_well >= 0 && idx_bore >= 0 )
			well_list[idx_well].wellborelist.erase(well_list[idx_well].wellborelist.begin()+idx_bore, well_list[idx_well].wellborelist.begin()+idx_bore+1);
	}

	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		int idx = getIndexFromVectorString(well_wellbore_basket, txt);
		if ( idx >= 0 )
		{
			well_wellbore_basket.erase(well_wellbore_basket.begin()+idx, well_wellbore_basket.begin()+idx+1);
			well_head_tinyname_basket.erase(well_head_tinyname_basket.begin()+idx, well_head_tinyname_basket.begin()+idx+1);
			well_head_fullname_basket.erase(well_head_fullname_basket.begin()+idx, well_head_fullname_basket.begin()+idx+1);
			well_bore_tinyname_basket.erase(well_bore_tinyname_basket.begin()+idx, well_bore_tinyname_basket.begin()+idx+1);
			well_bore_fullname_basket.erase(well_bore_fullname_basket.begin()+idx, well_bore_fullname_basket.begin()+idx+1);
		}
	}
	display_well_basket_list();

	this->lw_wells->clearSelection();
	qlw_welllog->clear();
	qlw_welltf2p->clear();
	qlw_wellpicks->clear();
	display_welllog_basket_list();
	display_welltf2p_basket_list();
	display_wellpicks_basket_list();
}

void GeotimeProjectManagerWidget::well_tf2p_clear() {
	for (int idx_well=0; idx_well<well_list.size(); idx_well++)
	{
		for (int idx_bore=0; idx_bore<well_list[idx_well].wellborelist.size(); idx_bore++)
		{
			WELLBORELIST& wellBoreList = well_list[idx_well].wellborelist[idx_bore];
			wellBoreList.tf2p_tinyname.erase(wellBoreList.tf2p_tinyname.begin(), wellBoreList.tf2p_tinyname.end());
			wellBoreList.tf2p_fullname.erase(wellBoreList.tf2p_fullname.begin(), wellBoreList.tf2p_fullname.end());
			wellBoreList.tf2p_displayname.erase(wellBoreList.tf2p_displayname.begin(), wellBoreList.tf2p_displayname.end());
		}
	}
	display_welltf2p_basket_list();
	this->qlw_welltf2p_basket->clearSelection();
}

void GeotimeProjectManagerWidget::display_well_basket_list()
{
	this->lw_wellsbasket->clear();
	int cpt = 0;
	for (int i=0; i<this->well_wellbore_basket.size(); i++)
	{
		QString txt = this->well_wellbore_basket[i];
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(QString(txt));
		item->setToolTip(QString(txt));
		this->lw_wellsbasket->addItem(item);
	}
}


void GeotimeProjectManagerWidget::trt_seismic_basket_add()
{
    seismic_basket_add();
    display_seismic_basket_list();
    this->lw_seismic->clearSelection();
}

void GeotimeProjectManagerWidget::trt_seismic_basket_sub()
{
    seismic_basket_sub();
    display_seismic_basket_list();
    this->lw_seismic_basket->clearSelection();
}

void GeotimeProjectManagerWidget::trt_nurbs_basket_add()
{
    nurbs_basket_add();
    display_nurbs_basket_list();
    this->lw_nurbs->clearSelection();
}

void GeotimeProjectManagerWidget::trt_nurbs_basket_sub()
{
    nurbs_basket_sub();
    display_nurbs_basket_list();
    this->lw_nurbs_basket->clearSelection();
}


void GeotimeProjectManagerWidget::trt_horizons_basket_add()
{
    horizons_basket_add();
    display_horizons_basket_list();
    this->lw_horizons->clearSelection();
}

void GeotimeProjectManagerWidget::trt_horizons_basket_sub()
{
    horizons_basket_sub();
    display_horizons_basket_list();
    this->lw_horizons_basket->clearSelection();
}



void GeotimeProjectManagerWidget::trt_rgb_basket_add()
{
    rgb_basket_add();
    display_rgb_basket_list();
    this->qlw_rgb->clearSelection();
}

void GeotimeProjectManagerWidget::trt_rgb_basket_sub()
{
    rgb_basket_sub();
    display_rgb_basket_list();
    this->qlw_rgb_basket->clearSelection();
}






void GeotimeProjectManagerWidget::clear_wells_data(int type)
{
	this->well_list.clear();

	lw_wells->clear();
	lw_wellsbasket->clear();
	lw_wellbore->clear();
	qlw_welllog->clear();
	qlw_welllog_basket->clear();
	qlw_welltf2p->clear();
	qlw_welltf2p_basket->clear();
	qlw_wellpicks->clear();
	qlw_wellpicks_basket->clear();

	wells_maindir_fullname.clear();
	wells_maindir_tinyname.clear();
	well_head_fullname.clear();
	well_head_tinyname.clear();
	well_bore_fullname.clear();
	well_bore_tinyname.clear();
	well_wellbore_tinyname.clear();
	well_bore.clear();
	well_head_tinyname_basket.clear();
	well_head_fullname_basket.clear();
	well_bore_tinyname_basket.clear();
	well_bore_fullname_basket.clear();
	well_wellbore_basket.clear();
	wells_fullname.clear();
	wells_tinyname.clear();
	wellslog_fullname.clear();
	wellslog_tinyname.clear();
	wellsft2p_fullname.clear();
	wellsft2p_tinyname.clear();
	wellspicks_fullname.clear();
	wellspicks_tinyname.clear();
	deviation_fullname.clear();
	deviation_tinyname.clear();
	wellslog_basket_fullname.clear();
	wellslog_basket_tinyname.clear();
	wellsft2p_basket_fullname.clear();
	wellsft2p_basket_tinyname.clear();
	wellspicks_basket_fullname.clear();
	wellspicks_basket_tinyname.clear();

	qlw_welllog->clear();
	qlw_welllog_basket->clear();
	qlw_welltf2p->clear();
	qlw_welltf2p_basket->clear();
	qlw_wellpicks->clear();
	qlw_wellpicks_basket->clear();
	wellslog_fullname.clear();
	wellslog_tinyname.clear();
	wellsft2p_fullname.clear();
	wellsft2p_tinyname.clear();
	wellspicks_fullname.clear();
	wellspicks_tinyname.clear();
	deviation_fullname.clear();
	deviation_tinyname.clear();
	wellslog_basket_fullname.clear();
	wellslog_basket_tinyname.clear();
	wellsft2p_basket_fullname.clear();
	wellsft2p_basket_tinyname.clear();
	wellspicks_basket_fullname.clear();
	wellspicks_basket_tinyname.clear();
	if ( type == 1 )
	{
		wells_fullname.clear();
		wells_tinyname.clear();
		wells_maindir_fullname.clear();
		wells_maindir_tinyname.clear();
		lw_wells->clear();
	}

}

void GeotimeProjectManagerWidget::welllog_names_update(QString path, int n_well, int n_bore)
{
	if ( !isTabWells() ) return;
	if ( n_well < 0 || n_bore < 0 ) return;

	// QString prefix = QString("[ ") + display0.wells[n_well].head_tinyname + " - " + display0.wells[n_well].bore[n_bore].bore_tinyname + QString(" ] - ");

	display0.wells[n_well].bore[n_bore].log_tinyname.clear();
	display0.wells[n_well].bore[n_bore].log_fullname.clear();

	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.log";
	dir.setNameFilters(filters);



	QFileInfoList list =getFiles(path.toStdString(),filters);// dir.entryInfoList();


	int N = list.size();
	std::vector<QString> seismic_list;
	display0.wells[n_well].bore[n_bore].log_tinyname.resize(N);
	display0.wells[n_well].bore[n_bore].log_fullname.resize(N);
	char buff[10000];
	for (int i=0; i<list.size(); i++)
	{
		QFileInfo fileInfo = list.at(i);
	    QString filename = fileInfo.fileName();
	    QString full_filename = fileInfo.absoluteFilePath();
	    FILE *pfile = fopen(full_filename.toStdString().c_str(), "r");
	    if ( pfile != NULL )
	    {
	    	int nn = 0;
	    	int cont = 1;

	    	while ( cont )
	    	{
	    		int nbre = fscanf(pfile, "Name\t%s\n", buff);
	    		if ( nbre > 0 )
	    			cont = 0;
	    		else
	    			fgets(buff, 10000, pfile);
	    		nn++;
	    		if ( nn > 10 ) { cont = 0; strcpy(buff, filename.toStdString().c_str()); }
	    	}
	    	fclose(pfile);
	    	display0.wells[n_well].bore[n_bore].log_tinyname[i] = QString(buff);
	    }
	    else
	    {
	    	display0.wells[n_well].bore[n_bore].log_tinyname[i] = filename;
	    }
	    display0.wells[n_well].bore[n_bore].log_fullname[i] = full_filename;
	}
}

void GeotimeProjectManagerWidget::welltf2p_names_update(QString path, int n_well, int n_bore)
{
	if ( !isTabWells() ) return;
	if ( n_well < 0 || n_bore < 0 ) return;

	display0.wells[n_well].bore[n_bore].tf2p_tinyname.clear();
	display0.wells[n_well].bore[n_bore].tf2p_fullname.clear();

	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.tfp";
	dir.setNameFilters(filters);

	QFileInfoList list =getFiles(path.toStdString(),filters); //dir.entryInfoList();
	int N = list.size();
	std::vector<QString> seismic_list;
	display0.wells[n_well].bore[n_bore].tf2p_tinyname.resize(N);
	display0.wells[n_well].bore[n_bore].tf2p_fullname.resize(N);
	char buff[10000];
	for (int i=0; i<list.size(); i++)
	{
		QFileInfo fileInfo = list.at(i);
		QString filename = fileInfo.fileName();
		QString full_filename = path + "/" + filename;
		FILE *pfile = fopen(full_filename.toStdString().c_str(), "r");
		if ( pfile != NULL )
		{
			int nn = 0;
		    int cont = 1;
	    	while ( cont )
	    	{
	    		int nbre = fscanf(pfile, "Name\t%s\n", buff);
	    		if ( nbre > 0 )
	    			cont = 0;
	    		else
	    			fgets(buff, 10000, pfile);
	    		nn++;
	    		if ( nn > 10 ) { cont = 0; strcpy(buff, filename.toStdString().c_str()); }
	    	}
	    	fclose(pfile);
	    	display0.wells[n_well].bore[n_bore].tf2p_tinyname[i] = QString(buff);
	    }
	    else
	    {
	    	display0.wells[n_well].bore[n_bore].tf2p_tinyname[i] = filename;
	    }
		display0.wells[n_well].bore[n_bore].tf2p_fullname[i] = full_filename;
	}
}

QFileInfoList GeotimeProjectManagerWidget::getFiles(std::string path, QStringList ext)
{
	QFileInfoList list;

	if(fs::exists(path))
	{

		for( const auto & entry : fs::directory_iterator(path))
		{
			for(int i=0;i<ext.size();i++)
			{

				QString newext = ext[i].replace("*","");
				if ( endsWith(entry.path().c_str(), newext.toStdString()) == 1)
				{
					if(fs::is_regular_file(entry.path().c_str()) )
					{
						list.append(QFileInfo(QString::fromStdString(entry.path().c_str())));
					}
				}

			}

		}
	}
	return list;
}


void GeotimeProjectManagerWidget::wellpicks_names_update(QString path, int n_well, int n_bore)
{
	if ( !isTabWells() ) return;
	if ( n_well < 0 || n_bore < 0 ) return;

	display0.wells[n_well].bore[n_bore].picks_tinyname.clear();
	display0.wells[n_well].bore[n_bore].picks_fullname.clear();

	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.pick";
	dir.setNameFilters(filters);

	qDebug()<<" wellpicks_names_update";
	QFileInfoList list = getFiles(path.toStdString(),QStringList(".pick"));

	/*for( const auto & entry : fs::directory_iterator(path.toStdString())) {
			//std::cout << entry.path() << std::endl;

			if ( endsWith(entry.path().c_str(), ".pick") == 1) {
				list.append(QFileInfo(QString::fromStdString(entry.path().c_str())));
				//m_descFile = entry.path().c_str();
				//readLayerPropertyDesc(entry.path().c_str());
			}

		}*/

	//QFileInfoList list = dir.entryInfoList();
	qDebug()<<" finwellpicks_names_update ==>"<<list.size();

	int N = list.size();
	std::vector<QString> seismic_list;
	display0.wells[n_well].bore[n_bore].picks_tinyname.resize(N);
	display0.wells[n_well].bore[n_bore].picks_fullname.resize(N);

	char buff[10000];
	for (int i=0; i<list.size(); i++)
	{
		QFileInfo fileInfo = list.at(i);
		QString filename = fileInfo.fileName();
		QString full_filename = path + "/" + filename;
		FILE *pfile = fopen(full_filename.toStdString().c_str(), "r");
		if ( pfile != NULL )
		{
			int nn = 0;
		    int cont = 1;
	    	while ( cont )
	    	{
	    		int nbre = fscanf(pfile, "Name\t%s\n", buff);
	    		if ( nbre > 0 )
	    			cont = 0;
	    		else
	    			fgets(buff, 10000, pfile);
	    		nn++;
	    		if ( nn > 10 ) { cont = 0; strcpy(buff, filename.toStdString().c_str()); }
	    	}
	    	fclose(pfile);
	    	display0.wells[n_well].bore[n_bore].picks_tinyname[i] = QString(buff);
	    }
	    else
	    {
	    	display0.wells[n_well].bore[n_bore].picks_tinyname[i] = filename;
	    }
		display0.wells[n_well].bore[n_bore].picks_fullname[i] = full_filename;
	}
}


void GeotimeProjectManagerWidget::welldeviation_names_update(QString path, int n_well, int n_bore)
{
	if ( !isTabWells() ) return;

	deviation_fullname.clear();
	QString filename = path + "/deviation";
	deviation_fullname.resize(1);
	if ( QFile::exists(filename) )
	{
		display0.wells[n_well].bore[n_bore].deviation_fullname = filename;
		deviation_fullname[0] = filename;
	}
	else
		display0.wells[n_well].bore[n_bore].deviation_fullname = QString("");
}

QString GeotimeProjectManagerWidget::get_deviation_fullname(QString wellbore_fullname) {
	QString deviation_fullname = wellbore_fullname + "/deviation";

	if ( !QFile::exists(deviation_fullname) ) {
		deviation_fullname = "";
	}


	return deviation_fullname;
}

/*
void GeotimeProjectManagerWidget::welldeviation_names_update(QString path)
{
	if ( !isTabWells() ) return;

	deviation_fullname.clear();

	QString filename = path + "/deviation";
	deviation_fullname.resize(1);
	if ( QFile::exists(filename) )
	{
		deviation_fullname[0] = filename;
	}
	else
		deviation_fullname[0] = QString("");
}
*/

void GeotimeProjectManagerWidget::welllog_names_update0(QString path)
{
	if ( !isTabWells() ) return;

	int idx_well = -1, idx_bore = -1;
	welllist_get_index_from_wellbore_fullname(path, &idx_well, &idx_bore);

	wellslog_fullname.clear();
	wellslog_tinyname.clear();

	int N = display0.wells[idx_well].bore[idx_bore].log_tinyname.size();
	wellslog_fullname.resize(N);
	wellslog_tinyname.resize(N);

	for (int i=0; i<N; i++)
	{
		wellslog_tinyname[i] = display0.wells[idx_well].bore[idx_bore].log_tinyname[i];
		wellslog_fullname[i] = display0.wells[idx_well].bore[idx_bore].log_fullname[i];
	}
}

void GeotimeProjectManagerWidget::welltf2p_names_update0(QString path)
{
	if ( !isTabWells() ) return;

	int idx_well = -1, idx_bore = -1;
	welllist_get_index_from_wellbore_fullname(path, &idx_well, &idx_bore);

	wellsft2p_fullname.clear();
	wellsft2p_tinyname.clear();

	int N = display0.wells[idx_well].bore[idx_bore].tf2p_tinyname.size();
	wellsft2p_fullname.resize(N);
	wellsft2p_tinyname.resize(N);
	for (int i=0; i<N; i++)
	{
		wellsft2p_tinyname[i] = display0.wells[idx_well].bore[idx_bore].tf2p_tinyname[i];
		wellsft2p_fullname[i] = display0.wells[idx_well].bore[idx_bore].tf2p_fullname[i];
	}
}

void GeotimeProjectManagerWidget::wellpicks_names_update0(QString path)
{
	if ( !isTabWells() ) return;

	int idx_well = -1, idx_bore = -1;
	welllist_get_index_from_wellbore_fullname(path, &idx_well, &idx_bore);

	wellspicks_fullname.clear();
	wellspicks_tinyname.clear();

	int N = display0.wells[idx_well].bore[idx_bore].picks_tinyname.size();
	wellspicks_fullname.resize(N);
	wellspicks_tinyname.resize(N);

	for (int i=0; i<N; i++)
	{
		wellspicks_tinyname[i] = display0.wells[idx_well].bore[idx_bore].picks_tinyname[i];
		wellspicks_fullname[i] = display0.wells[idx_well].bore[idx_bore].picks_fullname[i];
	}
}


void GeotimeProjectManagerWidget::welldeviation_names_update0(QString path)
{
	if ( !isTabWells() ) return;

	deviation_fullname.clear();

	QString filename = path + "/deviation";
	deviation_fullname.resize(1);
	if ( QFile::exists(filename) )
	{
		deviation_fullname[0] = filename;
	}
	else
		deviation_fullname[0] = QString("");
}

void GeotimeProjectManagerWidget::display_welllog(QString path, QString prefix)
{

	this->qlw_welllog->clear();
	QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
	int nbsearch = list1.size();

	for (int i=0; i<wellslog_tinyname.size(); i++)
	{
	    if ( qstring_display_valid(wellslog_tinyname[i], prefix) )
	    {
	    	QListWidgetItem *item = new QListWidgetItem;
	        item->setText(wellslog_tinyname[i]);
	        item->setToolTip(wellslog_tinyname[i]);
	        this->qlw_welllog->addItem(item);
	        // this->qlw_welllog->addItem(wellslog_tinyname[i]);
	    }
	}
}

void GeotimeProjectManagerWidget::display_welltf2p(QString path, QString prefix)
{
	this->qlw_welltf2p->clear();
	QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
	int nbsearch = list1.size();

	for (int i=0; i<wellsft2p_tinyname.size(); i++)
	{
		if ( qstring_display_valid(wellsft2p_tinyname[i], prefix) )
		{
			QListWidgetItem *item = new QListWidgetItem;
			item->setText(wellsft2p_tinyname[i]);
			item->setToolTip(wellsft2p_tinyname[i]);
			this->qlw_welltf2p->addItem(item);
	     }
	 }
}

void GeotimeProjectManagerWidget::display_wellpicks(QString path, QString prefix)
{

	this->qlw_wellpicks->clear();
	QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
	int nbsearch = list1.size();

	for (int i=0; i<wellspicks_tinyname.size(); i++)
	{
		if ( qstring_display_valid(wellspicks_tinyname[i], prefix) )
		{
			QListWidgetItem *item = new QListWidgetItem;
			item->setText(wellspicks_tinyname[i]);
			item->setToolTip(wellspicks_tinyname[i]);
			this->qlw_wellpicks->addItem(item);
		}
	}
}

void GeotimeProjectManagerWidget::display_welllog_basket_list()
{
	this->qlw_welllog_basket->clear();

	if ( this->lw_wellsbasket->currentItem() == NULL ) return;
	QString temp = this->lw_wellsbasket->currentItem()->text();
	int idx = getIndexFromVectorString(this->well_wellbore_basket, temp);
	if ( idx == -1 ) return;

	QString head_tinyname = well_head_tinyname_basket[idx];
	QString head_fullname = well_head_fullname_basket[idx];
	QString bore_tinyname = well_bore_tinyname_basket[idx];
	QString bore_fullname = well_bore_fullname_basket[idx];
	QString deviationfullname = this->deviation_fullname[0];

	int idx_well = -1, idx_bore = -1;
	welllist_create_get_index(head_tinyname, head_fullname, bore_tinyname, bore_fullname, deviationfullname, &idx_well, &idx_bore);
	if ( idx_well < 0 || idx_bore < 0 ) return;


	fprintf(stderr, "%d %d %d\n", well_list.size(), well_list[idx_well].wellborelist.size(), well_list[idx_well].wellborelist[idx_bore].log_displayname.size());

	for (int j=0; j<this->well_list[idx_well].wellborelist[idx_bore].log_displayname.size(); j++)
	{
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(well_list[idx_well].wellborelist[idx_bore].log_displayname[j]);
		item->setToolTip(well_list[idx_well].wellborelist[idx_bore].log_displayname[j]);
		this->qlw_welllog_basket->addItem(item);
		// this->qlw_welllog_basket->addItem(well_list[n].wellborelist[i].log_displayname[j]);
	}
	// qDebug() << deviationfullname;
}

void GeotimeProjectManagerWidget::display_welltf2p_basket_list()
{
	this->qlw_welltf2p_basket->clear();

	if ( this->lw_wellsbasket->currentItem() == NULL ) return;
	QString temp = this->lw_wellsbasket->currentItem()->text();
	int idx = getIndexFromVectorString(this->well_wellbore_basket, temp);
	if ( idx == -1 ) return;

	QString head_tinyname = well_head_tinyname_basket[idx];
	QString head_fullname = well_head_fullname_basket[idx];
	QString bore_tinyname = well_bore_tinyname_basket[idx];
	QString bore_fullname = well_bore_fullname_basket[idx];
	QString deviationfullname = this->deviation_fullname[0];

	int idx_well = -1, idx_bore = -1;
	welllist_create_get_index(head_tinyname, head_fullname, bore_tinyname, bore_fullname, deviationfullname, &idx_well, &idx_bore);
	if ( idx_well < 0 || idx_bore < 0 ) return;

	for (int j=0; j<this->well_list[idx_well].wellborelist[idx_bore].tf2p_displayname.size(); j++)
	{
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(well_list[idx_well].wellborelist[idx_bore].tf2p_displayname[j]);
		item->setToolTip(well_list[idx_well].wellborelist[idx_bore].tf2p_displayname[j]);
		this->qlw_welltf2p_basket->addItem(item);
		// this->qlw_welllog_basket->addItem(well_list[n].wellborelist[i].log_displayname[j]);
	}
}

void GeotimeProjectManagerWidget::display_wellpicks_basket_list()
{
	this->qlw_wellpicks_basket->clear();

	if ( this->lw_wellsbasket->currentItem() == NULL ) return;
	QString temp = this->lw_wellsbasket->currentItem()->text();
	int idx = getIndexFromVectorString(this->well_wellbore_basket, temp);
	if ( idx == -1 ) return;

	QString head_tinyname = well_head_tinyname_basket[idx];
	QString head_fullname = well_head_fullname_basket[idx];
	QString bore_tinyname = well_bore_tinyname_basket[idx];
	QString bore_fullname = well_bore_fullname_basket[idx];
	QString deviationfullname = this->deviation_fullname[0];

	int idx_well = -1, idx_bore = -1;
	welllist_create_get_index(head_tinyname, head_fullname, bore_tinyname, bore_fullname, deviationfullname, &idx_well, &idx_bore);
	if ( idx_well < 0 || idx_bore < 0 ) return;

	for (int j=0; j<this->well_list[idx_well].wellborelist[idx_bore].picks_displayname.size(); j++)
	{
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(well_list[idx_well].wellborelist[idx_bore].picks_displayname[j]);
		item->setToolTip(well_list[idx_well].wellborelist[idx_bore].picks_displayname[j]);
		this->qlw_wellpicks_basket->addItem(item);
		// this->qlw_welllog_basket->addItem(well_list[n].wellborelist[i].log_displayname[j]);
	}
}

void GeotimeProjectManagerWidget::trt_welllistclick(QListWidgetItem *p)
{

}


void GeotimeProjectManagerWidget::trt_welllogsearchchange(QString str)
{
	display_welllog("", linedit_welllogsearch->text());
}

void GeotimeProjectManagerWidget::trt_welltf2psearchchange(QString str)
{
	display_welltf2p("", linedit_welltf2psearch->text());
}

void GeotimeProjectManagerWidget::trt_wellpickssearchchchange(QString str)
{
	display_wellpicks("", linedit_wellpickssearch->text());
}


void GeotimeProjectManagerWidget::welllist_get_index_from_borename(QString name, int *idx_well, int *idx_bore)
{
	*idx_well = -1;
	*idx_bore = -1;
	for (int n=0; n<well_list.size(); n++)
	{
		for (int m=0; m<well_list[n].wellborelist.size(); m++)
		{
			if ( well_list[n].wellborelist[m].bore_fullname.compare(name) == 0 )
			{
				*idx_well = n;
				*idx_bore = m;
				return;
			}
		}
	}
}

void GeotimeProjectManagerWidget::welllist_create_get_index(QString head_tinyname, QString head_fullname,
		QString bore_tinyname, QString bore_fullname,
		QString deviation_fullname,
		int *idx_well, int *idx_bore)
{
	*idx_well = -1;
	*idx_bore = -1;
	int idx0 = -1, n = 0;
	while ( n < well_list.size() && idx0 < 0 )
	{
		// qDebug() << well_list[n].head_fullname << head_fullname;

		if ( well_list[n].head_fullname.compare(head_fullname) == 0 )
		{
			idx0 = n;
		}
		n++;
	}

	if ( idx0 < 0 )
	{
		WELLLIST welllist;
		well_list.push_back(welllist);
		int n0 = well_list.size()-1;
		well_list[n0].head_tinyname = head_tinyname;
		well_list[n0].head_fullname = head_fullname;
		*idx_well = n0;
	}
	else
	{
		*idx_well = idx0;
	}

	idx0 = -1, n = 0;
	while ( n < well_list[*idx_well].wellborelist.size() && idx0 < 0 )
	{
		if ( well_list[*idx_well].wellborelist[n].bore_fullname.compare(bore_fullname) == 0 )
		{
			idx0 = n;
		}
		n++;
	}
	if ( idx0 < 0 )
	{
		WELLBORELIST wellborelist;
		well_list[*idx_well].wellborelist.push_back(wellborelist);
		int n0 = well_list[*idx_well].wellborelist.size()-1;
		well_list[*idx_well].wellborelist[n0].bore_tinyname = bore_tinyname;
		well_list[*idx_well].wellborelist[n0].bore_fullname = bore_fullname;
		well_list[*idx_well].wellborelist[n0].deviation_fullname = deviation_fullname;
		*idx_bore = n0;
	}
	else
	{
		*idx_bore = idx0;
	}
}

void GeotimeProjectManagerWidget::welllist_get_index(QString head_tinyname, QString head_fullname,
		QString bore_tinyname, QString bore_fullname,
		int *idx_well, int *idx_bore)
{
	*idx_well = -1;
	*idx_bore = -1;
	int idx0 = -1, n = 0;
	while ( n < well_list.size() && idx0 < 0 )
	{
		if ( well_list[n].head_fullname.compare(head_fullname) == 0 )
		{
			idx0 = n;
		}
		n++;
	}

	if ( idx0 < 0 ) return;

	*idx_well = idx0;
	idx0 = -1, n = 0;
	while ( n < well_list[*idx_well].wellborelist.size() && idx0 < 0 )
	{
		if ( well_list[*idx_well].wellborelist[n].bore_fullname.compare(bore_fullname) == 0 )
		{
			idx0 = n;
		}
		n++;
	}
	if ( idx0 < 0 )
	{
		*idx_well = -1;
		*idx_bore = -1;
		return;
	}
	*idx_bore = idx0;
}

void GeotimeProjectManagerWidget::welllist_get_index_from_logname(QString log_displayname, int *idx_well, int *idx_bore, int *idx)
{
	*idx_well = -1;
	*idx_bore = -1;
	*idx = -1;

	QString name = getWellsHeadBasketSelectedName();
	if ( name.isEmpty() ) return;

	for (int n=0; n<well_list.size(); n++)
	{
		for (int m=0; m<well_list[n].wellborelist.size(); m++)
		{
			if ( name.compare(well_list[n].wellborelist[m].bore_tinyname) == 0 )
			{
				for (int p=0; p<well_list[n].wellborelist[m].log_displayname.size(); p++)
				{
					if ( well_list[n].wellborelist[m].log_displayname[p].compare(log_displayname) == 0 )
					{
						*idx_well = n;
						*idx_bore = m;
						*idx = p;
						return;
					}
				}
			}
		}
	}
}

void GeotimeProjectManagerWidget::welllist_get_index_from_tf2pname(QString displayname, int *idx_well, int *idx_bore, int *idx)
{
	*idx_well = -1;
	*idx_bore = -1;
	*idx = -1;

	QString name = getWellsHeadBasketSelectedName();
	if ( name.isEmpty() ) return;

	for (int n=0; n<well_list.size(); n++)
	{
		for (int m=0; m<well_list[n].wellborelist.size(); m++)
		{
			if ( name.compare(well_list[n].wellborelist[m].bore_tinyname) == 0 )
			{
				for (int p=0; p<well_list[n].wellborelist[m].tf2p_displayname.size(); p++)
				{
					if ( well_list[n].wellborelist[m].tf2p_displayname[p].compare(displayname) == 0 )
					{
						*idx_well = n;
						*idx_bore = m;
						*idx = p;
						return;
					}
				}
			}
		}
	}
}

void GeotimeProjectManagerWidget::welllist_get_index_from_picksname(QString displayname, int *idx_well, int *idx_bore, int *idx)
{
	*idx_well = -1;
	*idx_bore = -1;
	*idx = -1;

	QString name = getWellsHeadBasketSelectedName();
	if ( name.isEmpty() ) return;

	for (int n=0; n<well_list.size(); n++)
	{
		for (int m=0; m<well_list[n].wellborelist.size(); m++)
		{
			if ( name.compare(well_list[n].wellborelist[m].bore_tinyname) == 0 )
			{
				for (int p=0; p<well_list[n].wellborelist[m].picks_displayname.size(); p++)
				{
					if ( well_list[n].wellborelist[m].picks_displayname[p].compare(displayname) == 0 )
					{
						*idx_well = n;
						*idx_bore = m;
						*idx = p;
						return;
					}
				}
			}
		}
	}
}

void GeotimeProjectManagerWidget::welllist_get_index_from_wellbore_fullname(QString fullname, int *idx_well, int *idx_bore)
{
	*idx_well = -1;
	*idx_bore = -1;

	for (int n=0; n<display0.wells.size(); n++)
	{
		for (int m=0; m<display0.wells[n].bore.size(); m++)
		{

				if ( display0.wells[n].bore[m].bore_fullname.compare(fullname) == 0 )
				{
					*idx_well = n;
					*idx_bore = m;
					return;
				}
		}
	}
}





void GeotimeProjectManagerWidget::trt_welllog_basket_add()
{

	QList<QListWidgetItem*> list0 = qlw_welllog->selectedItems();
	if ( list0.size() == 0 ) return;

	if ( this->lw_wellsbasket->currentItem() == NULL ) return;
	QString temp = this->lw_wellsbasket->currentItem()->text();
	int idx = getIndexFromVectorString(this->well_wellbore_basket, temp);
	if ( idx == -1 ) return;

	QString head_tinyname = well_head_tinyname_basket[idx];
	QString head_fullname = well_head_fullname_basket[idx];
	QString bore_tinyname = well_bore_tinyname_basket[idx];
	QString bore_fullname = well_bore_fullname_basket[idx];
	QString deviationfullname = this->deviation_fullname[0];

	// fprintf(stderr, "head: [%s] %s\n", head_tinyname.toStdString().c_str(), head_fullname.toStdString().c_str());
	// fprintf(stderr, "bore: [%s] %s\n", bore_tinyname.toStdString().c_str(), bore_fullname.toStdString().c_str());

	int idx_well = -1, idx_bore = -1;
	welllist_create_get_index(head_tinyname, head_fullname, bore_tinyname, bore_fullname, deviationfullname, &idx_well, &idx_bore);

	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		int idx = getIndexFromVectorString(wellslog_tinyname, txt);
		if ( idx >= 0 )
		{
			int idx_basket = getIndexFromVectorString(well_list[idx_well].wellborelist[idx_bore].log_fullname, wellslog_fullname[idx]);
			if ( idx_basket < 0 )
			{
				well_list[idx_well].wellborelist[idx_bore].log_tinyname.push_back(wellslog_tinyname[idx]);
				well_list[idx_well].wellborelist[idx_bore].log_fullname.push_back(wellslog_fullname[idx]);
				// QString string = QString("[ ") + head_tinyname + QString( " - ") + bore_tinyname + QString(" ] ");
				QString string = wellslog_tinyname[idx];
				well_list[idx_well].wellborelist[idx_bore].log_displayname.push_back(string);
				// fprintf(stderr, "*** %s %s\n", wellslog_tinyname[idx].toStdString().c_str(), wellslog_fullname[idx].toStdString().c_str());
			}
		}
	}
	display_welllog_basket_list();
	this->qlw_welllog->clearSelection();
	// qDebug() << deviationfullname;
}

void GeotimeProjectManagerWidget::trt_welllog_basket_sub()
{
	QList<QListWidgetItem*> list0 = qlw_welllog_basket->selectedItems();
	if ( list0.size() == 0 ) return;


	 for (int i=0; i<list0.size(); i++)
	 {
		 QString txt = list0[i]->text();
		 int idx_well = -1;
		 int idx_bore = -1;
		 int  idx = -1;
		 welllist_get_index_from_logname(txt, &idx_well, &idx_bore, &idx);
	     if ( idx_well >= 0 && idx_bore >= 0 )
	     {
	    	 // int idx = getIndexFromVectorString(well_list[idx_well].wellborelist[idx_bore].log_displayname, txt);
	    	 well_list[idx_well].wellborelist[idx_bore].log_tinyname.erase(well_list[idx_well].wellborelist[idx_bore].log_tinyname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].log_tinyname.begin()+idx+1);
	         well_list[idx_well].wellborelist[idx_bore].log_fullname.erase(well_list[idx_well].wellborelist[idx_bore].log_fullname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].log_fullname.begin()+idx+1);
	         well_list[idx_well].wellborelist[idx_bore].log_displayname.erase(well_list[idx_well].wellborelist[idx_bore].log_displayname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].log_displayname.begin()+idx+1);
	     }
	 }
	 display_welllog_basket_list();
	 this->qlw_welllog_basket->clearSelection();
}

void GeotimeProjectManagerWidget::trt_welltf2p_basket_add()
{
	QList<QListWidgetItem*> list0 = qlw_welltf2p->selectedItems();
	if ( list0.size() == 0 ) return;

	QString temp = this->lw_wellsbasket->currentItem()->text();
	int idx = getIndexFromVectorString(this->well_wellbore_basket, temp);
	if ( idx == -1 ) return;

	QString head_tinyname = well_head_tinyname_basket[idx];
	QString head_fullname = well_head_fullname_basket[idx];
	QString bore_tinyname = well_bore_tinyname_basket[idx];
	QString bore_fullname = well_bore_fullname_basket[idx];
	QString deviationfullname = this->deviation_fullname[0];

	int idx_well = -1, idx_bore = -1;
	welllist_create_get_index(head_tinyname, head_fullname, bore_tinyname, bore_fullname, deviationfullname, &idx_well, &idx_bore);

	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		int idx = getIndexFromVectorString(wellsft2p_tinyname, txt);
		if ( idx >= 0 )
		{
			int idx_basket = getIndexFromVectorString(well_list[idx_well].wellborelist[idx_bore].tf2p_fullname, wellsft2p_fullname[idx]);
			if ( idx_basket < 0 )
			{
				well_list[idx_well].wellborelist[idx_bore].tf2p_tinyname.push_back(wellsft2p_tinyname[idx]);
				well_list[idx_well].wellborelist[idx_bore].tf2p_fullname.push_back(wellsft2p_fullname[idx]);
				QString string = wellsft2p_tinyname[idx];
				well_list[idx_well].wellborelist[idx_bore].tf2p_displayname.push_back(string);
				// fprintf(stderr, "*** %s %s\n", wellslog_tinyname[idx].toStdString().c_str(), wellslog_fullname[idx].toStdString().c_str());
			}
		}
	}
	display_welltf2p_basket_list();
	this->qlw_welltf2p->clearSelection();
}

void GeotimeProjectManagerWidget::trt_welltf2p_basket_sub()
{
	QList<QListWidgetItem*> list0 = qlw_welltf2p_basket->selectedItems();
	if ( list0.size() == 0 ) return;

	for (int i=0; i<list0.size(); i++)
	{
		 QString txt = list0[i]->text();
		 int idx_well = -1;
		 int idx_bore = -1;
		 int  idx = -1;
		 welllist_get_index_from_tf2pname(txt, &idx_well, &idx_bore, &idx);
	     if ( idx_well >= 0 && idx_bore >= 0 )
	     {
	    	 well_list[idx_well].wellborelist[idx_bore].tf2p_tinyname.erase(well_list[idx_well].wellborelist[idx_bore].tf2p_tinyname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].tf2p_tinyname.begin()+idx+1);
	    	 well_list[idx_well].wellborelist[idx_bore].tf2p_fullname.erase(well_list[idx_well].wellborelist[idx_bore].tf2p_fullname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].tf2p_fullname.begin()+idx+1);
	    	 well_list[idx_well].wellborelist[idx_bore].tf2p_displayname.erase(well_list[idx_well].wellborelist[idx_bore].tf2p_displayname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].tf2p_displayname.begin()+idx+1);
	     }
	}
	display_welltf2p_basket_list();
	this->qlw_welltf2p_basket->clearSelection();
}

void GeotimeProjectManagerWidget::trt_wellpicks_basket_add()
{
	QList<QListWidgetItem*> list0 = qlw_wellpicks->selectedItems();
	if ( list0.size() == 0 ) return;

	QString temp = this->lw_wellsbasket->currentItem()->text();
	int idx = getIndexFromVectorString(this->well_wellbore_basket, temp);
	if ( idx == -1 ) return;

	QString head_tinyname = well_head_tinyname_basket[idx];
	QString head_fullname = well_head_fullname_basket[idx];
	QString bore_tinyname = well_bore_tinyname_basket[idx];
	QString bore_fullname = well_bore_fullname_basket[idx];
	QString deviationfullname = this->deviation_fullname[0];

	int idx_well = -1, idx_bore = -1;
	welllist_create_get_index(head_tinyname, head_fullname, bore_tinyname, bore_fullname, deviationfullname, &idx_well, &idx_bore);
	for (int i=0; i<list0.size(); i++)
	{
		QString txt = list0[i]->text();
		int idx = getIndexFromVectorString(wellspicks_tinyname, txt);
		if ( idx >= 0 )
		{
			int idx_basket = getIndexFromVectorString(well_list[idx_well].wellborelist[idx_bore].picks_fullname,
					wellspicks_fullname[idx]);
			if ( idx_basket < 0 )
			{
				well_list[idx_well].wellborelist[idx_bore].picks_tinyname.push_back(wellspicks_tinyname[idx]);
				well_list[idx_well].wellborelist[idx_bore].picks_fullname.push_back(wellspicks_fullname[idx]);
				QString string = wellspicks_tinyname[idx];
				well_list[idx_well].wellborelist[idx_bore].picks_displayname.push_back(string);
			}
		}
	}
	display_wellpicks_basket_list();
	this->qlw_wellpicks->clearSelection();
}

void GeotimeProjectManagerWidget::trt_wellpicks_basket_sub()
{
	QList<QListWidgetItem*> list0 = qlw_wellpicks_basket->selectedItems();
	if ( list0.size() == 0 ) return;

	for (int i=0; i<list0.size(); i++)
	{
		 QString txt = list0[i]->text();
		 int idx_well = -1;
		 int idx_bore = -1;
		 int  idx = -1;
		 welllist_get_index_from_picksname(txt, &idx_well, &idx_bore, &idx);
	     if ( idx_well >= 0 && idx_bore >= 0 )
	     {
	    	 well_list[idx_well].wellborelist[idx_bore].picks_tinyname.erase(well_list[idx_well].wellborelist[idx_bore].picks_tinyname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].picks_tinyname.begin()+idx+1);
	    	 well_list[idx_well].wellborelist[idx_bore].picks_fullname.erase(well_list[idx_well].wellborelist[idx_bore].picks_fullname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].picks_fullname.begin()+idx+1);
	    	 well_list[idx_well].wellborelist[idx_bore].picks_displayname.erase(well_list[idx_well].wellborelist[idx_bore].picks_displayname.begin()+idx, well_list[idx_well].wellborelist[idx_bore].picks_displayname.begin()+idx+1);
	     }
	}
	display_wellpicks_basket_list();
	this->qlw_wellpicks_basket->clearSelection();
}


void GeotimeProjectManagerWidget::trt_cultural_database_update()
{
	cultural_names_disk_update();
	cultural_names_database_create();
    display_cultural_list(lineedit_wellssearch->text());
}

void GeotimeProjectManagerWidget::well_database_update()
{
	// wells_names_disk_update();
	// wells_names_database_create();

	QString path = get_wells_path0();
	QString db_filename = get_well_database_name();
	WellsDatabaseManager::update(db_filename, path);
	wells_names_update();
	display_wells_list(lineedit_wellssearch->text());
}

void GeotimeProjectManagerWidget::pick_database_update() {
	m_picksManager->trt_dataBaseUpdate();
}


void GeotimeProjectManagerWidget::trt_well_database_update()
{
	well_database_update();
}

void GeotimeProjectManagerWidget::trt_horizon_database_update()
{
	horizon_names_disk_update();
	horizon_names_database_create();
	display_horizons_list(lineedit_horizonssearch->text());
}


void GeotimeProjectManagerWidget::seismic_database_update()
{
	QString filename = get_seismic_database_name();
	QString datasetPath = get_seismic_path0();
	SeismisDatabaseManager::update(filename, datasetPath);
	// seismic_names_disk_update();
	// seismic_names_database_create();
	seismic_names_update();
	display_seismic_list(lineedit_seismicsearch->text());
}

void GeotimeProjectManagerWidget::trt_seismic_database_update()
{
	seismic_database_update();
}

void GeotimeProjectManagerWidget::trt_nurbs_database_update()
{
	nurbs_names_disk_update();
//	seismic_names_database_create();
	display_nurbs_list(lineedit_nurbssearch->text());
}


void GeotimeProjectManagerWidget::global_seismic_database_update()
{
	seismic_names_disk_update();
	seismic_names_database_create();
	display_seismic_list(lineedit_seismicsearch->text());
}

void GeotimeProjectManagerWidget::global_rgb_database_update()
{
	rgb_names_disk_update();
	// seismic_names_database_create();
	display_rgb_list(lineedit_rgt2rgtsearch->text());
}


// session

typedef struct _QSTRING_PAIR
{
	QString filename1;
	QString filename2;
}QSTRING_PAIR;

static QSTRING_PAIR get_pairfilenames(QJsonObject wellObj)
{
	QSTRING_PAIR out;

	return out;
}

void GeotimeProjectManagerWidget::load_session(QString sessionPath) {
    QFile file(sessionPath);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "GeotimeProjectManagerWidget : cannot load session, file not readable";
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject()) {
        qDebug() << "GeotimeProjectManagerWidget : cannot load session, root is not a json object";
        return;
    }

    bool isProjectValid = false;
    bool isSurveyValid = false;
    memoryRAZ();

    QJsonObject rootObj = doc.object();
    if (rootObj.contains(projectTypeKey)) {
    	int idx = 0;
    	QString project_type = rootObj.value(projectTypeKey).toString("None");

    	if ( project_type.compare("USER") == 0 && rootObj.contains(projectPathKey) )
    	{
    		QString projectPath = rootObj.value(projectPathKey).toString(".");
    		lineedit_custompath->setText(projectPath);
    	}

    	while(idx<cb_projecttype->count() && project_type.compare(cb_projecttype->itemText(idx))!=0) {
    		idx ++;
    	}
    	if (idx>=cb_projecttype->count()) {
    		idx = 0; // set as None
    	}

    	// avoid to call uneeded slots
    	{
			QSignalBlocker block(lineedit_projectsearch);
			lineedit_projectsearch->setText(""); // clear filter to avoid filtering the loaded project

			QSignalBlocker blockWell(lineedit_wellssearch);
			lineedit_wellssearch->setText("");

			QSignalBlocker blockCultural(lineedit_culturalsearch);
			lineedit_culturalsearch->setText("");

			QSignalBlocker blockNeuron(lineedit_neuronssearch);
			lineedit_neuronssearch->setText("");

			QSignalBlocker blockWellBore(linedit_wellboresearch);
			linedit_wellboresearch->setText("");

			QSignalBlocker blockLog(linedit_welllogsearch);
			linedit_welllogsearch->setText("");

			QSignalBlocker blockTFP(linedit_welltf2psearch);
			linedit_welltf2psearch->setText("");

			QSignalBlocker blockPick(linedit_wellpickssearch);
			linedit_wellpickssearch->setText("");
    	}

    	cb_projecttype->setCurrentIndex(idx); //trt_projecttypeclick(idx); // load project type

    	if (idx!=0 && rootObj.contains(projectKey)) { // load project
    		QString project = rootObj.value(projectKey).toString("");

    		std::size_t projectIdx = 0;
    		while(projectIdx<lw_projetlist->count() && project.compare(lw_projetlist->item(projectIdx)->text())!=0) {
    			projectIdx++;
    		}
    		if (projectIdx<lw_projetlist->count()) {
    			disconnect(lineedit_surveysearch, SIGNAL(textChanged(QString)), this, SLOT(trt_surveysearchchange(QString)));
    			lineedit_surveysearch->setText("");
    			connect(lineedit_surveysearch, SIGNAL(textChanged(QString)), this, SLOT(trt_surveysearchchange(QString)));

    			//lw_projetlist->item(projectIdx)->setSelected(true);
    			lw_projetlist->setCurrentItem(lw_projetlist->item(projectIdx));
    			trt_projetlistClick(lw_projetlist->item(projectIdx));

    			isProjectValid = true;
    		}
    	}

		if (isProjectValid && rootObj.contains(surveyKey)) {
			QString survey = rootObj.value(surveyKey).toString("");
			std::size_t surveyIdx = 0;
			while(surveyIdx<lw_surveylist->count() && survey.compare(lw_surveylist->item(surveyIdx)->text())!=0) {
				surveyIdx++;
			}
			if (surveyIdx<lw_surveylist->count()) {
				lw_surveylist->setCurrentItem(lw_surveylist->item(surveyIdx));
				trt_surveylistClick(lw_surveylist->item(surveyIdx));
				this->qlw_wellpicks->clearSelection();
				isSurveyValid = true;
			}
		}

    	display_label_titles();
    }

    // load culturals, wells and neurons
    if (isProjectValid) {

    	if (rootObj.contains(culturalKey) && rootObj.value(culturalKey).isArray()) {
    		QJsonArray array = rootObj.value(culturalKey).toArray();
    		for (std::size_t i=0; i<array.count(); i++) {
    			std::size_t searchIdx = 0;
    			QString txt = array[i].toString("");
    			while (searchIdx<lw_cultural->count() && txt.compare(lw_cultural->item(searchIdx)->text())!=0) {
    				searchIdx++;
    			}
    			if (searchIdx<lw_cultural->count()) {
    				lw_cultural->item(searchIdx)->setSelected(true);
    			}
    		}
    	}


    	// PICKS
    	if (rootObj.contains(picksNamesKey) && rootObj.value(picksNamesKey).isArray() &&
    			rootObj.contains(picksPathKey) && rootObj.value(picksPathKey).isArray()) {
    		std::vector<QString> tiny;
    		std::vector<QString> full;
    		QJsonArray arrayNames = rootObj.value(picksNamesKey).toArray();
    		QJsonArray arrayPath = rootObj.value(picksPathKey).toArray();
    		for (std::size_t i=0; i<arrayNames.count(); i++) {
    			tiny.push_back(arrayNames[i].toString(""));
    			full.push_back(arrayPath[i].toString(""));
    			ProjectManagerNames p;
    			p.copy(tiny, full);
    			m_picksManager->setBasketNames(p);
    			m_picksManager->displayNamesBasket();
    		}
    	}


    	if (rootObj.contains(wellKey) && rootObj.value(wellKey).isArray())
    	{
    		QJsonArray array = rootObj.value(wellKey).toArray();

    		for (std::size_t i=0; i<array.count(); i++)
    		{
    			if (!array[i].isObject())
    			{
    				continue;
    			}
    			QJsonObject wellObj = array[i].toObject();
    			if (!wellObj.contains(wellKey) || !wellObj.value(wellKey).isString() || !wellObj.contains(wellLogKey) || !wellObj.value(wellLogKey).isArray())
    			{
    				continue;
    			}


    			std::size_t searchIdx = 0;
    			if (!wellObj.contains(wellPathKey) || !wellObj.value(wellPathKey).isString()) {
    				QString txt = wellObj.value(wellKey).toString("");

    				while (searchIdx<lw_wells->count() && txt.compare(lw_wells->item(searchIdx)->text())!=0)
    				{
    					searchIdx++;
    				}
    			} else {
    				QString txt = wellObj.value(wellPathKey).toString("");

    				while (searchIdx<lw_wells->count() && txt.compare(lw_wells->item(searchIdx)->data(Qt::UserRole).toString())!=0)
    				{
    					searchIdx++;
    				}
    			}
    			if (searchIdx<lw_wells->count())
    			{
    				lw_wells->item(searchIdx)->setSelected(true);
    				trt_well_basket_add();
    			}

    		}


    		well_tf2p_clear();
    		for (std::size_t i=0; i<array.count(); i++)
    		{
    			if (!array[i].isObject())
    			{
    				continue;
    			}
    			QJsonObject wellObj = array[i].toObject();
    			QString wellbore = wellObj[wellKey].toString();



    			QJsonArray welllog = wellObj[wellLogKey].toArray();
    			// qDebug() << wellbore ;
    			int N = welllog.size();
    			// QStringList list = wellbore.split(" ");
    			// QString wellname_tinyname = list[1];
    			// QString borename_tinyname = list[3];

    			for (int ii=0; ii<N; ii++)
    			{
    				QJsonObject obj1 = welllog[ii].toObject();
    				int idx_well = -1, idx_bore = -1;
    				QString log_fullname = obj1["fullname"].toString();
    				QString log_tinyname = obj1["tinyname"].toString();

    				int idx0 = log_fullname.lastIndexOf("/");
    				QString borefullname = log_fullname.left(idx0);
    				idx0 = borefullname.lastIndexOf("/");
    				QString wellfullname = borefullname.left(idx0);

    				QString deviationfullname = borefullname + "/deviation";
    				// qDebug() << wellfullname << borefullname;
    				welllist_create_get_index(wellfullname, wellfullname, borefullname, borefullname, deviationfullname, &idx_well, &idx_bore);

    				well_list[idx_well].wellborelist[idx_bore].log_tinyname.push_back(log_tinyname);
    				well_list[idx_well].wellborelist[idx_bore].log_fullname.push_back(log_fullname);
    				well_list[idx_well].wellborelist[idx_bore].log_displayname.push_back(log_tinyname);
    			}



    			QJsonArray welltfp= wellObj[wellTFPKey].toArray();
    			// qDebug() << wellbore ;
    			N = welltfp.size();
    			for (int ii=0; ii<N; ii++)
    			{
    				QJsonObject obj1 = welltfp[ii].toObject();
    				int idx_well = -1, idx_bore = -1;
    				QString tfp_fullname = obj1["fullname"].toString();
    				QString tfp_tinyname = obj1["tinyname"].toString();

    				int idx0 = tfp_fullname.lastIndexOf("/");
    				QString borefullname = tfp_fullname.left(idx0);
    				idx0 = borefullname.lastIndexOf("/");
    				QString wellfullname = borefullname.left(idx0);

    				QString deviationfullname = borefullname + "/deviation";
    				// qDebug() << wellfullname << borefullname;
    				welllist_create_get_index(wellfullname, wellfullname, borefullname, borefullname, deviationfullname, &idx_well, &idx_bore);

    				well_list[idx_well].wellborelist[idx_bore].tf2p_tinyname.push_back(tfp_tinyname);
    				well_list[idx_well].wellborelist[idx_bore].tf2p_fullname.push_back(tfp_fullname);
    				well_list[idx_well].wellborelist[idx_bore].tf2p_displayname.push_back(tfp_tinyname);
    			}

//    			QJsonArray wellpicks = wellObj[wellPickKey].toArray();
//    			qDebug() << "picks" << wellbore << wellObj[wellPickKey].isArray() << wellpicks.size();
//    			N = wellpicks.size();
//    			for (int ii=0; ii<N; ii++)
//    			{
//    				QJsonObject obj1 = wellpicks[ii].toObject();
//    				int idx_well = -1, idx_bore = -1;
//    				QString picks_fullname = obj1["fullname"].toString();
//    				QString picks_tinyname = obj1["tinyname"].toString();
//
//    				int idx0 = picks_fullname.lastIndexOf("/");
//    			    QString borefullname = picks_fullname.left(idx0);
//    			    idx0 = borefullname.lastIndexOf("/");
//    			    QString wellfullname = borefullname.left(idx0);
//
//    			    QString deviationfullname = borefullname + "/deviation";
//    			    // qDebug() << wellfullname << borefullname;
//    			    welllist_create_get_index(wellfullname, wellfullname, borefullname, borefullname, deviationfullname, &idx_well, &idx_bore);
//
//    			    well_list[idx_well].wellborelist[idx_bore].picks_tinyname.push_back(picks_tinyname);
//    			    well_list[idx_well].wellborelist[idx_bore].picks_fullname.push_back(picks_fullname);
//    			    well_list[idx_well].wellborelist[idx_bore].picks_displayname.push_back(picks_tinyname);
//    			}
    		}
    	}

    	if (rootObj.contains(neuronKey) && rootObj.value(neuronKey).isArray()) {
    		QJsonArray array = rootObj.value(neuronKey).toArray();
    		for (std::size_t i=0; i<array.count(); i++) {
    			std::size_t searchIdx = 0;
    			QString txt = array[i].toString("");
    			while (searchIdx<lw_neurons->count() && txt.compare(lw_neurons->item(searchIdx)->text())!=0) {
    				searchIdx++;
    			}
    			if (searchIdx<lw_neurons->count()) {
    				lw_neurons->item(searchIdx)->setSelected(true);
    			}
    		}
    	}
    }


    // load seismic and horizons
    if (isSurveyValid) {
    	if (rootObj.contains(seismicKey) && rootObj.value(seismicKey).isArray()) {
    		QJsonArray array = rootObj.value(seismicKey).toArray();
    		seismic_tinyname_basket.clear();
    		m_seismic_fullpath_basket.clear();
    		seismic_tinyname_basket.reserve(array.count());
    		m_seismic_fullpath_basket.reserve(array.count());
    	    for (int i=0; i<array.count(); i++)
    	    {
    	        QString txt = array[i].toString("");
    	        int idx = getIndexFromVectorString(seismic_tinyname, txt);
    	        if ( idx >= 0 )
    	        {
    	            seismic_tinyname_basket.push_back(seismic_tinyname[idx]);
    	            m_seismic_fullpath_basket.push_back(m_seismic_fullpath[idx]);
    	            seismic_basket_color.push_back(seismic_diplay_color[idx]);
    	        }
    	    }
    	}
    	display_seismic_basket_list();

    	if (rootObj.contains(horizonKey) && rootObj.value(horizonKey).isArray()) {
    		QJsonArray array = rootObj.value(horizonKey).toArray();
			freehorizons_tinyname_basket.clear();
			freehorizons_fullname_basket.clear();
			freehorizons_tinyname_basket.reserve(array.count());
			freehorizons_fullname_basket.reserve(array.count());

			std::vector<QString> freehorizons_all_fullnames = get_freehorizon_fullnames();
			for (int i=0; i<array.count(); i++)
			{
				if (array[i].isObject())
				{
					QJsonObject obj = array[i].toObject();
					if (obj.contains(tinynameKey) && obj.value(tinynameKey).isString() &&
							obj.contains(fullnameKey) && obj.value(fullnameKey).isString())
					{
						int idx = getIndexFromVectorString(freehorizons_all_fullnames, obj.value(fullnameKey).toString(""));
						if (idx>=0)
						{
							freehorizons_tinyname_basket.push_back(obj.value(tinynameKey).toString(""));
							freehorizons_fullname_basket.push_back(obj.value(fullnameKey).toString(""));
						}
					}
				}
			}
    	}
    	//display_horizons_basket_list();

    	if (rootObj.contains(isoHorizonKey) && rootObj.value(isoHorizonKey).isArray()) {
			QJsonArray array = rootObj.value(isoHorizonKey).toArray();
			isohorizons_tinyname_basket.clear();
			isohorizons_fullname_basket.clear();
			isohorizons_tinyname_basket.reserve(array.count());
			isohorizons_fullname_basket.reserve(array.count());

			std::vector<QString> isohorizons_all_fullnames = get_isohorizon_fullnames();
			for (int i=0; i<array.count(); i++)
			{
				if (array[i].isObject())
				{
					QJsonObject obj = array[i].toObject();
					if (obj.contains(tinynameKey) && obj.value(tinynameKey).isString() &&
							obj.contains(fullnameKey) && obj.value(fullnameKey).isString())
					{
						int idx = getIndexFromVectorString(isohorizons_all_fullnames, obj.value(fullnameKey).toString(""));
						if (idx>=0)
						{
							isohorizons_tinyname_basket.push_back(obj.value(tinynameKey).toString(""));
							isohorizons_fullname_basket.push_back(obj.value(fullnameKey).toString(""));
						}
					}
				}
			}
		}

		if (rootObj.contains(nurbsKey) && rootObj.value(nurbsKey).isArray()) {
			QJsonArray array = rootObj.value(nurbsKey).toArray();
			nurbs_tinyname_basket.clear();
			nurbs_fullname_basket.clear();
			nurbs_tinyname_basket.reserve(array.count());
			nurbs_fullname_basket.reserve(array.count());
			for (int i=0; i<array.count(); i++)
			{
				if (array[i].isObject())
				{
					QJsonObject obj = array[i].toObject();
					if (obj.contains(tinynameKey) && obj.value(tinynameKey).isString() &&
							obj.contains(fullnameKey) && obj.value(fullnameKey).isString())
					{
						int idx = getIndexFromVectorString(nurbs_fullname, obj.value(fullnameKey).toString(""));
						if (idx>=0)
						{
							nurbs_tinyname_basket.push_back(obj.value(tinynameKey).toString(""));
							nurbs_fullname_basket.push_back(obj.value(fullnameKey).toString(""));
						}
					}
				}
			}
		}
		display_nurbs_basket_list();

		if (rootObj.contains(horizonAnimKey) && rootObj.value(horizonAnimKey).isArray()) {
			QJsonArray array = rootObj.value(horizonAnimKey).toArray();
			horizonanims_tinyname_basket.clear();
			horizonanims_fullname_basket.clear();
			horizonanims_tinyname_basket.reserve(array.count());
			horizonanims_fullname_basket.reserve(array.count());

			std::vector<QString> horizonanims_all_fullnames = get_horizonanim_fullnames0();
			for (int i=0; i<array.count(); i++)
			{
				if (array[i].isObject())
				{
					QJsonObject obj = array[i].toObject();
					if (obj.contains(tinynameKey) && obj.value(tinynameKey).isString() &&
							obj.contains(fullnameKey) && obj.value(fullnameKey).isString())
					{
						int idx = getIndexFromVectorString(horizonanims_all_fullnames, obj.value(fullnameKey).toString(""));
						if (idx>=0)
						{
							horizonanims_tinyname_basket.push_back(obj.value(tinynameKey).toString(""));
							horizonanims_fullname_basket.push_back(obj.value(fullnameKey).toString(""));
						}
					}
				}
			}
		}

    }
    display_welllog_basket_list();


    display_nurbs_basket_list();

}

void GeotimeProjectManagerWidget::save_session(QString sessionPath) {
	QFile file(sessionPath);
	if (!file.open(QIODevice::WriteOnly)) {
		qDebug() << "GeotimeProjectManagerWidget : cannot save session, file not writable";
		return;
	}

	QJsonObject obj;
	obj.insert(projectTypeKey, cb_projecttype->currentText());
	if ( cb_projecttype->currentText().compare("USER") == 0 )
	{
		obj.insert(projectPathKey, lineedit_custompath->text());
	}

	if (cb_projecttype->currentText().compare("None")!=0 && lw_projetlist->currentItem()!=nullptr) {

		obj.insert(projectKey, lw_projetlist->currentItem()->text());

		if (lw_surveylist->currentItem()!=nullptr) {
			obj.insert(surveyKey, lw_surveylist->currentItem()->text());

			QJsonArray seismics, horizons, culturals, wells, neurons, picksNames, picksPath;
			for (std::size_t arrayIdx=0; arrayIdx<lw_seismic_basket->count(); arrayIdx++) {
				QListWidgetItem* seismicItem = lw_seismic_basket->item(arrayIdx);
				seismics.append(seismicItem->text());
			}
			for (std::size_t arrayIdx=0; arrayIdx<freehorizons_fullname_basket.size(); arrayIdx++) {
				QJsonObject freeHorizonObj;
				freeHorizonObj[tinynameKey] = freehorizons_tinyname_basket[arrayIdx];
				freeHorizonObj[fullnameKey] = freehorizons_fullname_basket[arrayIdx];
				horizons.append(freeHorizonObj);
			}
			QJsonArray isoHorizons;
			for (std::size_t arrayIdx=0; arrayIdx<isohorizons_fullname_basket.size(); arrayIdx++) {
				QJsonObject isoHorizonObj;
				isoHorizonObj[tinynameKey] = isohorizons_tinyname_basket[arrayIdx];
				isoHorizonObj[fullnameKey] = isohorizons_fullname_basket[arrayIdx];
				isoHorizons.append(isoHorizonObj);
			}
			for (std::size_t arrayIdx=0; arrayIdx<lw_cultural->count(); arrayIdx++) {
				QListWidgetItem* culturalItem = lw_cultural->item(arrayIdx);
				if (culturalItem->isSelected()) {
					culturals.append(culturalItem->text());
				}
			}
			// picks new tab
			std::vector<QString> _picksNames = m_picksManager->getNames();
			for (std::size_t arrayIdx=0; arrayIdx<_picksNames.size(); arrayIdx++) {
				picksNames.append(_picksNames[arrayIdx]);
			}
			std::vector<QString> _picksPath = m_picksManager->getPath();
			for (std::size_t arrayIdx=0; arrayIdx<_picksPath.size(); arrayIdx++) {
				picksPath.append(_picksPath[arrayIdx]);
			}
			QJsonArray nurbs;
			for (std::size_t arrayIdx=0; arrayIdx<nurbs_fullname_basket.size(); arrayIdx++) {
				QJsonObject nurbsObj;
				nurbsObj[tinynameKey] = nurbs_tinyname_basket[arrayIdx];
				nurbsObj[fullnameKey] = nurbs_fullname_basket[arrayIdx];
				nurbs.append(nurbsObj);
			}
			QJsonArray horizonAnims;
			for (std::size_t arrayIdx=0; arrayIdx<horizonanims_fullname_basket.size(); arrayIdx++) {
				QJsonObject horizonAnimObj;
				horizonAnimObj[tinynameKey] = horizonanims_tinyname_basket[arrayIdx];
				horizonAnimObj[fullnameKey] = horizonanims_fullname_basket[arrayIdx];
				horizonAnims.append(horizonAnimObj);
			}


			for (std::size_t arrayIdx=0; arrayIdx<well_wellbore_basket.size(); arrayIdx++) {
				//QListWidgetItem* wellItem = lw_wellsbasket->item(arrayIdx);

				QString wellBoreUIName = well_wellbore_basket[arrayIdx];

//		    	well_wellbore_basket.push_back(well_wellbore_tinyname[idx]);
//		    	well_head_tinyname_basket.push_back(well_head_tinyname[idx]);
//		    	well_head_fullname_basket.push_back(well_head_fullname[idx]);
//		    	well_bore_tinyname_basket.push_back(well_bore_tinyname[idx]);
//		    	well_bore_fullname_basket.push_back(well_bore_fullname[idx]);

				QJsonObject wellObj;
				wellObj.insert(wellKey, wellBoreUIName);
				wellObj.insert(wellPathKey, well_bore_fullname_basket[arrayIdx]);

				// find well head and well bore indexes
				long idx_well = 0;
				long idx_bore = 0;
				while (idx_well<well_list.size() && well_head_fullname_basket[arrayIdx].compare(well_list[idx_well].head_fullname)!=0) {
					idx_well++;
				}
				if (idx_well>=well_list.size()) {
					qDebug() << "Save Session -- save well head : Inconsistent well " << wellBoreUIName; // Should not happen
					continue;
				}
				while (idx_bore<well_list[idx_well].wellborelist.size() && well_bore_fullname_basket[arrayIdx].compare(
						well_list[idx_well].wellborelist[idx_bore].bore_fullname)!=0) {
					idx_bore++;
				}
				if (idx_bore>=well_list[idx_well].wellborelist.size()) {
					qDebug() << "Save Session -- save well bore : Inconsistent well " << wellBoreUIName; // Should not happen
					continue;
				}

				QJsonArray logsArray, tfpArray, picksArray;

				const std::vector<QString>& log_tinyname = well_list[idx_well].wellborelist[idx_bore].log_tinyname;
				const std::vector<QString>& log_fullname = well_list[idx_well].wellborelist[idx_bore].log_fullname;
				for (long i=0; i<log_tinyname.size(); i++) {
					QJsonObject logObj;
					logObj.insert(tinynameKey, log_tinyname[i]);
					logObj.insert(fullnameKey, log_fullname[i]);
					logsArray.append(logObj);
				}
				wellObj.insert(wellLogKey, logsArray);

				const std::vector<QString>& tfp_tinyname = well_list[idx_well].wellborelist[idx_bore].tf2p_tinyname;
				const std::vector<QString>& tfp_fullname = well_list[idx_well].wellborelist[idx_bore].tf2p_fullname;
				for (long i=0; i<tfp_tinyname.size(); i++) {
					QJsonObject tfpObj;
					tfpObj.insert(tinynameKey, tfp_tinyname[i]);
					tfpObj.insert(fullnameKey, tfp_fullname[i]);
					tfpArray.append(tfpObj);
				}
				wellObj.insert(wellTFPKey, tfpArray);

//				const std::vector<QString>& pick_tinyname = well_list[idx_well].wellborelist[idx_bore].picks_tinyname;
//				const std::vector<QString>& pick_fullname = well_list[idx_well].wellborelist[idx_bore].picks_fullname;
//				for (long i=0; i<pick_tinyname.size(); i++) {
//					QJsonObject pickObj;
//					pickObj.insert(tinynameKey, pick_tinyname[i]);
//					pickObj.insert(fullnameKey, pick_fullname[i]);
//					picksArray.append(pickObj);
//				}
//				wellObj.insert(wellPickKey, picksArray);

				// TODO TFPs and picks

				wells.append(wellObj);

			}
			for (std::size_t arrayIdx=0; arrayIdx<lw_neurons->count(); arrayIdx++) {
				QListWidgetItem* neuronItem = lw_neurons->item(arrayIdx);
				if (neuronItem->isSelected()) {
					neurons.append(neuronItem->text());
				}
			}
			obj.insert(seismicKey, seismics);
			obj.insert(horizonKey, horizons);
			obj.insert(isoHorizonKey, isoHorizons);
			obj.insert(culturalKey, culturals);
			obj.insert(wellKey, wells);
			obj.insert(neuronKey, neurons);
			obj.insert(picksNamesKey, picksNames);
			obj.insert(picksPathKey, picksPath);
			obj.insert(nurbsKey, nurbs);
			obj.insert(horizonAnimKey, horizonAnims);
		}
	}


    QJsonDocument doc(obj);
    file.write(doc.toJson());
}

void GeotimeProjectManagerWidget::load_session_gui() {
    QSettings settings;
    GlobalConfig& config = GlobalConfig::getConfig();
    const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, config.sessionPath()).toString();

    const QString filePath = QFileDialog::getOpenFileName(this,
                                                          tr("Load Session"),
                                                          lastPath,
                                                          QLatin1String("*.json"));
    if (filePath.isEmpty())
        return;

    const QFileInfo fi(filePath);
    settings.setValue(LAST_SESSION_PATH_IN_SETTINGS, fi.absoluteFilePath());

    load_session(filePath);
	init_session_cache();
}

void GeotimeProjectManagerWidget::save_session_gui() {
    QSettings settings;
    GlobalConfig& config = GlobalConfig::getConfig();
    QString defaultSessionPath = config.sessionPath();
    if (!QFileInfo(defaultSessionPath).exists()) {
        QDir dir = QFileInfo(defaultSessionPath).absoluteDir();
        dir.mkdir(QFileInfo(defaultSessionPath).fileName());
    }

    const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, defaultSessionPath).toString();

    QFileDialog fileDialog(this, tr("Save Session"), lastPath, QLatin1String("*.json"));
    fileDialog.setDefaultSuffix("json");
    fileDialog.setAcceptMode(QFileDialog::AcceptSave);
    int result = fileDialog.exec();

    QString filePath;
    if (result==QDialog::Accepted && fileDialog.selectedFiles().size()>0) {
    	filePath = fileDialog.selectedFiles()[0];
    } else {
        return;
    }

    const QFileInfo fi(filePath);
    settings.setValue(LAST_SESSION_PATH_IN_SETTINGS, fi.absoluteFilePath());

    save_session(filePath);
	init_session_cache();
}

void GeotimeProjectManagerWidget::load_last_session() {
	QSettings settings;
	const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, "").toString();
	if (lastPath.isEmpty() || !QFileInfo(lastPath).exists())
		return;

	load_session(lastPath);
	init_session_cache();
}

void GeotimeProjectManagerWidget::init_session_cache() {
	QSettings settings;
	const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, "").toString();

	bool sessionPathValid = !lastPath.isNull() && !lastPath.isEmpty() && QFileInfo(lastPath).exists() && QFileInfo(lastPath).isFile();
	if (sessionPathValid) {
		m_session_path_cache = lastPath;
		m_session_loaded = true;
		m_session_project_type_cache = cb_projecttype->currentText();
		m_session_use_path_cache = lineedit_custompath->text();
		if (lw_projetlist->currentItem()) {
			m_session_project_cache = lw_projetlist->currentItem()->text();
		}
		if (lw_surveylist->currentItem()) {
			m_session_survey_cache = lw_surveylist->currentItem()->text();
		}
	} else {
		m_session_loaded = false;
		m_session_path_cache = "";
	}
}

bool GeotimeProjectManagerWidget::is_session_loaded() {
	bool loaded = m_session_loaded && m_session_project_type_cache.compare(cb_projecttype->currentText())==0;

	if (loaded && cb_projecttype->currentText().compare("USER")==0) {
		loaded = m_session_use_path_cache.compare(lineedit_custompath->text())==0;
	}
	loaded = loaded && lw_projetlist->currentItem()!=nullptr &&
			m_session_project_cache.compare(lw_projetlist->currentItem()->text())==0 &&
			lw_surveylist->currentItem()!=nullptr &&
			m_session_survey_cache.compare(lw_surveylist->currentItem()->text())==0;
	return loaded;
}

bool GeotimeProjectManagerWidget::save_to_default_session(bool force) {
	if (!force && !is_session_loaded()) {
		return false;
	}

	bool doSave = !m_session_path_cache.isNull() && !m_session_path_cache.isEmpty() &&
			QFileInfo(m_session_path_cache).exists() && QFileInfo(m_session_path_cache).isFile();
	if (doSave) {
		save_session(m_session_path_cache);
	}
	return doSave;
}

// picks sorted wells
std::vector<MARKER> GeotimeProjectManagerWidget::getPicksSortedWells()
{
	std::vector<QString> picksName = m_picksManager->getNames();
	std::vector<QBrush> colors = m_picksManager->getColors();
	std::vector<PMANAGER_WELL_DISPLAY> wellsList0 = get_display_well_list();


	std::vector<MARKER> data;
	int N = picksName.size();
	data.resize(N);
	int Nwells = wellsList0.size();
	for (int n=0; n<N; n++)
	{
		QString pname = picksName[n];
		data[n].name = pname;
		data[n].color = colors[n].color();
		for (int n2=0; n2<Nwells; n2++)
		{
			WELLPICKSLIST wellList;
			wellList.name = wellsList0[n2].head_tinyname;;
			wellList.path = wellsList0[n2].head_fullname;;

			for (int nb=0; nb<wellsList0[n2].bore.size(); nb++)
			{
				std::vector<QString> tiny = wellsList0[n2].bore[nb].picks_tinyname;
				std::vector<QString> full = wellsList0[n2].bore[nb].picks_fullname;
				for (int i=0; i<tiny.size(); i++)
				{
					if ( pname.compare(tiny[i]) == 0 )
					{
						WELLBOREPICKSLIST wellBore;
						wellBore.boreName = wellsList0[n2].bore[nb].bore_tinyname;;
						wellBore.borePath = wellsList0[n2].bore[nb].bore_fullname;;
						wellBore.deviationPath = wellBore.borePath + "/deviation";
						wellBore.picksName = tiny[i];
						wellBore.picksPath = full[i];
						wellList.wellBore.push_back(wellBore);
					}
				}
			}
			if ( wellList.wellBore.size() > 0 )
				data[n].wellPickLists.push_back(wellList);
		}
	}
	return data;
}

// picks sorted wells
std::vector<MARKER> GeotimeProjectManagerWidget::staticGetPicksSortedWells(const std::vector<QString>& picksName,
		const std::vector<QBrush>& colors, const std::vector<PMANAGER_WELL_DISPLAY>& wellsList0)
{
	std::vector<MARKER> data;
	int N = picksName.size();
	data.resize(N);
	int Nwells = wellsList0.size();
	for (int n=0; n<N; n++)
	{
		QString pname = picksName[n];
		data[n].name = pname;
		data[n].color = colors[n].color();
		for (int n2=0; n2<Nwells; n2++)
		{
			WELLPICKSLIST wellList;
			wellList.name = wellsList0[n2].head_tinyname;;
			wellList.path = wellsList0[n2].head_fullname;;

			for (int nb=0; nb<wellsList0[n2].bore.size(); nb++)
			{
				std::vector<QString> tiny = wellsList0[n2].bore[nb].picks_tinyname;
				std::vector<QString> full = wellsList0[n2].bore[nb].picks_fullname;
				for (int i=0; i<tiny.size(); i++)
				{
					if ( pname.compare(tiny[i]) == 0 )
					{
						WELLBOREPICKSLIST wellBore;
						wellBore.boreName = wellsList0[n2].bore[nb].bore_tinyname;;
						wellBore.borePath = wellsList0[n2].bore[nb].bore_fullname;;
						wellBore.deviationPath = wellBore.borePath + "/deviation";
						wellBore.picksName = tiny[i];
						wellBore.picksPath = full[i];
						wellList.wellBore.push_back(wellBore);
					}
				}
			}
			if ( wellList.wellBore.size() > 0 )
				data[n].wellPickLists.push_back(wellList);
		}
	}
	return data;
}

void GeotimeProjectManagerWidget::trt_debug()
{
	qDebug() << "DEBUG";

	std::vector<MARKER> data = getPicksSortedWells();
	char *filename = "/data/PLI/NKDEEP/jacques/wells0.txt";
	FILE *pf = fopen(filename, "w");
	int N = data.size();
	for (int n=0; n<N; n++)
	{
		for (int ii=0; ii<10; ii++)
			fprintf(pf, "=================== PICKS ==================\n");

		int N2 = data[n].wellPickLists.size();
		for (int n2=0; n2<N2; n2++)
		{
			fprintf(pf, "well: %d\n%s\n%s\n", n2, data[n].wellPickLists[n2].name.toStdString().c_str(), data[n].wellPickLists[n2].path.toStdString().c_str());
			std::vector<WELLBOREPICKSLIST> wellBore = data[n].wellPickLists[n2].wellBore;
			for (int i=0; i<wellBore.size(); i++)
			{
				fprintf(pf, "[%d]\n", i);
				fprintf(pf, "%s\n", wellBore[i].boreName.toStdString().c_str());
				fprintf(pf, "%s\n", wellBore[i].borePath.toStdString().c_str());
				fprintf(pf, "%s\n", wellBore[i].deviationPath.toStdString().c_str());
				fprintf(pf, "%s\n", wellBore[i].picksName.toStdString().c_str());
				fprintf(pf, "%s\n", wellBore[i].picksPath.toStdString().c_str());
				fprintf(pf, "=========================================\n");
			}
			fprintf(pf, "\n\n");
		}
	}
	fclose(pf);
}

std::vector<QString> GeotimeProjectManagerWidget::get_picks_names() {
	return m_picksManager->getNames();
}

std::vector<QString> GeotimeProjectManagerWidget::get_picks_fullnames() {
	return m_picksManager->getPath();
}

std::vector<QBrush> GeotimeProjectManagerWidget::get_picks_colors() {
	return m_picksManager->getColors();
}

std::vector<QString> GeotimeProjectManagerWidget::get_all_picks_names() {
	return m_picksManager->getAllNames();
}

std::vector<QString> GeotimeProjectManagerWidget::get_all_picks_fullnames() {
	return m_picksManager->getAllPath();
}

std::vector<QBrush> GeotimeProjectManagerWidget::get_all_picks_colors() {
	return m_picksManager->getAllColors();
}

void GeotimeProjectManagerWidget::clear_project_specific_lists() {
	// wells
	clear_wells_data(1); // may be too much

	// picks
	m_picksManager->dataBasketClear();
	m_picksManager->dataClear();
	m_picksManager->displayClear();
	m_picksManager->displayBasketClear();

	// culturals
    data0.culturals.cdata_tinyname.clear();
    data0.culturals.cdata_fullname.clear();
    data0.culturals.strd_tinyname.clear();
    data0.culturals.strd_fullname.clear();
	lw_cultural->clearSelection();
    display_culturals_basket_list();

	// neurons
    lw_neurons->clear();

	// survey
	clear_survey_specific_lists();
}

void GeotimeProjectManagerWidget::clear_survey_specific_lists() {
	// seismics
	seismic_fullname.clear();
	seismic_tinyname.clear();
	m_seismic_fullpath.clear();
	seismic_diplay_color.clear();
	seismic_tinyname_basket.clear();
	m_seismic_fullpath_basket.clear();
	seismic_basket_color.clear();

	// gui seismics
    display_seismic_basket_list();
    this->lw_seismic->clear();

	//horizons
	horizons_fullname.clear();
	horizons_tinyname.clear();
	horizons_tinyname_basket.clear();
	horizons_fullname_basket.clear();

	// gui horizons
    display_horizons_basket_list();
    this->lw_horizons->clear();

    // rgt rgb
    rgb_tinyname.clear();
    rgb_fullname.clear();
    rgb_basket_tinyname.clear();
    rgb_basket_fullname.clear();

    // gui rgb rgt
    display_rgb_basket_list();
    this->qlw_rgb->clear();

    nurbs_fullname.clear();
    nurbs_tinyname.clear();

    nurbs_tinyname_basket.clear();
    nurbs_fullname_basket.clear();

    freehorizons_tinyname_basket.clear();
    freehorizons_fullname_basket.clear();
    isohorizons_tinyname_basket.clear();
    isohorizons_fullname_basket.clear();
    horizonanims_tinyname_basket.clear();
    horizonanims_fullname_basket.clear();


	// gui seismics
	display_nurbs_basket_list();
	this->lw_nurbs->clear();


}

QString GeotimeProjectManagerWidget::format_dir_path(const QString& path_to_format) {
	// to remove redundant /, . and ..
	QDir dir = QDir(path_to_format);
	QString absolute_path = dir.absolutePath();

	// add missing / at the end
	if (absolute_path.count()>0 && absolute_path.back()!='/') {
		absolute_path += "/";
	}

	return absolute_path;
}

std::vector<QString> GeotimeProjectManagerWidget::get_nurbs_names0()
{
	return get_nurbs_list(get_nurbs_path0());
}

std::vector<QString> GeotimeProjectManagerWidget::get_nurbs_fullnames0()
{
	QString path = get_nurbs_path0();

	std::vector<QString>  list = get_nurbs_list(get_nurbs_path0());

	for(int i=0;i<list.size();i++)
	{
		list[i] = path+list[i];
	}

	return list;
}

std::vector<QString> GeotimeProjectManagerWidget::get_horizonanim_names0()
{
	return get_horizonanim_list(get_horizonanim_path0());
}

std::vector<QString> GeotimeProjectManagerWidget::get_horizonanim_fullnames0()
{
	QString path = get_horizonanim_path0();

	std::vector<QString>  list = get_horizonanim_list(get_horizonanim_path0());

	for(int i=0;i<list.size();i++)
	{
		list[i] = path+list[i];
	}

	return list;
}


std::vector<QString> GeotimeProjectManagerWidget::get_freehorizon_names()
{
	std::vector<QString> out;
	QString path = get_survey_fullpath_name() + "ImportExport/IJK/HORIZONS/" + QString::fromStdString(FreeHorizonManager::OldBaseDirectory) + "/";
	QFileInfoList infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].baseName());

	path = get_survey_fullpath_name() + "/" + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_PATH) + "/";
	infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].baseName());
	return out;
}


std::vector<QString> GeotimeProjectManagerWidget::get_freehorizon_fullnames()
{
	std::vector<QString> out;
	QString path = get_survey_fullpath_name() + "ImportExport/IJK/HORIZONS/" + QString::fromStdString(FreeHorizonManager::OldBaseDirectory) + "/";
	QFileInfoList infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].absoluteFilePath());

	path = get_survey_fullpath_name() + "/" + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_PATH) + "/";
	infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].absoluteFilePath());
	return out;
}

std::vector<QString> GeotimeProjectManagerWidget::get_isohorizon_names()
{
	std::vector<QString> out;
	QString path = get_survey_fullpath_name() + "ImportExport/IJK/HORIZONS/ISOVAL/";
	QFileInfoList infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].baseName());
	// new path
	path = get_survey_fullpath_name() + "/" + QString::fromStdString(GeotimePath::NEXTVISION_ISOHORIZON_PATH) + "/";
	infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].baseName());
	return out;
}

std::vector<QString> GeotimeProjectManagerWidget::get_isohorizon_fullnames()
{
	std::vector<QString> out;
	QString path = get_survey_fullpath_name() + "ImportExport/IJK/HORIZONS/ISOVAL/";
	QFileInfoList infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].absoluteFilePath());
	// new path
	path = get_survey_fullpath_name() + "/" + QString::fromStdString(GeotimePath::NEXTVISION_ISOHORIZON_PATH) + "/";
	infoList = QDir(path).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
	for (int i=0; i<infoList.size(); i++)
		out.push_back(infoList[i].absoluteFilePath());

	return out;
}

void GeotimeProjectManagerWidget::set_wells_filter(const QString& filter)
{
	lineedit_wellssearch->setText(filter);
}

bool GeotimeProjectManagerWidget::add_pick_fullpath_name(const QString& pick_fullpath_name)
{
	std::vector<QString> tinyAll = get_all_picks_names();
	std::vector<QString> fullAll = get_all_picks_fullnames();

	auto fullAllIt = std::find(fullAll.begin(), fullAll.end(), pick_fullpath_name);

	bool success = false;
	if (fullAllIt!=fullAll.end())
	{
		long idxAll = std::distance(fullAll.begin(), fullAllIt);

		std::vector<QString> tiny = get_picks_names();
		std::vector<QString> full = get_picks_fullnames();

		auto fullIt = std::find(full.begin(), full.end(), pick_fullpath_name);
		if (fullIt==full.end())
		{
			tiny.push_back(tinyAll[idxAll]);
			full.push_back(fullAll[idxAll]);

			ProjectManagerNames p;
			p.copy(tiny, full);
			m_picksManager->setBasketNames(p);
			m_picksManager->displayNamesBasket();

			success = true;
		}
	}
	return success;
}

bool GeotimeProjectManagerWidget::remove_pick_fullpath_name(const QString& pick_fullpath_name)
{
	std::vector<QString> tiny = get_picks_names();
	std::vector<QString> full = get_picks_fullnames();

	auto fullIt = std::find(full.begin(), full.end(), pick_fullpath_name);

	bool success = false;
	if (fullIt!=full.end())
	{
		long idxAll = std::distance(full.begin(), fullIt);
		if (idxAll<tiny.size())
		{
			auto tinyIt = tiny.begin();
			std::advance(tinyIt, idxAll);

			tiny.erase(tinyIt);
			full.erase(fullIt);

			ProjectManagerNames p;
			p.copy(tiny, full);
			m_picksManager->setBasketNames(p);
			m_picksManager->displayNamesBasket();

			success = true;
		}
	}
	return success;
}

void GeotimeProjectManagerWidget::set_seismics_filter(const QString& filter) {
	lineedit_seismicsearch->setText(filter);
}

bool GeotimeProjectManagerWidget::select_seismic_tinyname(const QString& searched_tinyname) {
	bool found = false;
	long i=0;
	while (!found && i<lw_seismic->count())
	{
		QListWidgetItem* item = lw_seismic->item(i);
		QString tinyname = item->text();

		found = searched_tinyname.compare(tinyname)==0;
		if (!found)
		{
			i++;
		}
	}
	if (found)
	{
		QListWidgetItem* item = lw_seismic->item(i);
		item->setSelected(true);
	}
	return found;
}

void GeotimeProjectManagerWidget::clear_seismic_gui_selection() {
	lw_seismic->clearSelection();
}

bool GeotimeProjectManagerWidget::select_seismic_basket_tinyname(const QString& searched_tinyname) {
	bool found = false;
	long i=0;
	while (!found && i<lw_seismic_basket->count())
	{
		QListWidgetItem* item = lw_seismic_basket->item(i);
		QString tinyname = item->text();

		found = searched_tinyname.compare(tinyname)==0;
		if (!found)
		{
			i++;
		}
	}
	if (found)
	{
		QListWidgetItem* item = lw_seismic_basket->item(i);
		item->setSelected(true);
	}
	return found;
}

void GeotimeProjectManagerWidget::clear_seismic_basket_gui_selection() {
	lw_seismic_basket->clearSelection();
}

void GeotimeProjectManagerWidget::set_tfps_filter(const QString& filter) {
	linedit_welltf2psearch->setText(filter);
}

bool GeotimeProjectManagerWidget::select_well_basket_tinyname(const QString& well_name) {
	bool foundOne = false;
	for (long i=0; i<lw_wellsbasket->count(); i++) {
		QListWidgetItem* item = lw_wellsbasket->item(i);
		if (item->text().compare(well_name)==0) {
			lw_wellsbasket->setCurrentItem(item, QItemSelectionModel::ClearAndSelect);
			foundOne = true;
		}
	}
	return foundOne;
}

void GeotimeProjectManagerWidget::select_all_tfp_basket() {
	qlw_welltf2p_basket->selectAll();
}

bool GeotimeProjectManagerWidget::select_tfp_tinyname(const QString& tfp_name) {
	bool foundOne = false;
	for (long i=0; i<qlw_welltf2p->count(); i++) {
		QListWidgetItem* item = qlw_welltf2p->item(i);
		if (item->text().compare(tfp_name)==0) {
			item->setSelected(true);
			foundOne = true;
		}
	}
	return foundOne;
}

void GeotimeProjectManagerWidget::add_freehorizon(const QString& freehorizon_fullname,
		const QString& freehorizon_tinyname) {
	freehorizons_fullname_basket.push_back(freehorizon_fullname);
	freehorizons_tinyname_basket.push_back(freehorizon_tinyname);
}

void GeotimeProjectManagerWidget::remove_freehorizon(const QString& freehorizon_fullname) {
	int i=0;
	while (i<freehorizons_fullname_basket.size() && freehorizon_fullname!=freehorizons_fullname_basket[i]) {
		i++;
	}
	if (i<freehorizons_fullname_basket.size()) {
		std::vector<QString>::iterator itFull = freehorizons_fullname_basket.begin();
		std::advance(itFull, i);
		std::vector<QString>::iterator itTiny = freehorizons_tinyname_basket.begin();
		std::advance(itTiny, i);
		freehorizons_tinyname_basket.erase(itTiny);
		freehorizons_fullname_basket.erase(itFull);
	}
}

std::vector<QString> GeotimeProjectManagerWidget::get_freehorizon_names_basket() {
	return freehorizons_tinyname_basket;
}

std::vector<QString> GeotimeProjectManagerWidget::get_freehorizon_fullnames_basket() {
	return freehorizons_fullname_basket;
}

void GeotimeProjectManagerWidget::add_isohorizon(const QString& isohorizon_fullname,
		const QString& isohorizon_tinyname) {
	isohorizons_fullname_basket.push_back(isohorizon_fullname);
	isohorizons_tinyname_basket.push_back(isohorizon_tinyname);
}

void GeotimeProjectManagerWidget::remove_isohorizon(const QString& isohorizon_fullname) {
	int i=0;
	while (i<isohorizons_fullname_basket.size() && isohorizon_fullname!=isohorizons_fullname_basket[i]) {
		i++;
	}
	if (i<isohorizons_fullname_basket.size()) {
		std::vector<QString>::iterator itFull = isohorizons_fullname_basket.begin();
		std::advance(itFull, i);
		std::vector<QString>::iterator itTiny = isohorizons_tinyname_basket.begin();
		std::advance(itTiny, i);
		isohorizons_tinyname_basket.erase(itTiny);
		isohorizons_fullname_basket.erase(itFull);
	}
}

std::vector<QString> GeotimeProjectManagerWidget::get_isohorizon_names_basket() {
	return isohorizons_tinyname_basket;
}

std::vector<QString> GeotimeProjectManagerWidget::get_isohorizon_fullnames_basket() {
	return isohorizons_fullname_basket;
}

void GeotimeProjectManagerWidget::add_horizonanim(const QString& horizonanim_fullname,
		const QString& horizonanim_tinyname) {
	horizonanims_fullname_basket.push_back(horizonanim_fullname);
	horizonanims_tinyname_basket.push_back(horizonanim_tinyname);
}

void GeotimeProjectManagerWidget::remove_horizonanim(const QString& horizonanim_fullname) {
	int i=0;
	while (i<horizonanims_fullname_basket.size() && horizonanim_fullname!=horizonanims_fullname_basket[i]) {
		i++;
	}
	if (i<horizonanims_fullname_basket.size()) {
		std::vector<QString>::iterator itFull = horizonanims_fullname_basket.begin();
		std::advance(itFull, i);
		std::vector<QString>::iterator itTiny = horizonanims_tinyname_basket.begin();
		std::advance(itTiny, i);
		horizonanims_tinyname_basket.erase(itTiny);
		horizonanims_fullname_basket.erase(itFull);
	}
}

std::vector<QString> GeotimeProjectManagerWidget::get_horizonanim_names_basket() {
	return horizonanims_tinyname_basket;
}

std::vector<QString> GeotimeProjectManagerWidget::get_horizonanim_fullnames_basket() {
	return horizonanims_fullname_basket;
}

void GeotimeProjectManagerWidget::set_nurbs_filter(const QString& filter) {
	lineedit_nurbssearch->setText(filter);
}

bool GeotimeProjectManagerWidget::select_nurbs_tinyname(const QString& searched_tinyname) {
	bool found = false;
	long i=0;
	while (!found && i<lw_nurbs->count())
	{
		QListWidgetItem* item = lw_nurbs->item(i);
		QString tinyname = item->text();

		found = searched_tinyname.compare(tinyname)==0;
		if (!found)
		{
			i++;
		}
	}
	if (found)
	{
		QListWidgetItem* item = lw_nurbs->item(i);
		item->setSelected(true);
	}
	return found;
}

void GeotimeProjectManagerWidget::clear_nurbs_gui_selection() {
	lw_nurbs->clearSelection();
}

bool GeotimeProjectManagerWidget::select_nurbs_basket_tinyname(const QString& searched_tinyname) {
	bool found = false;
	long i=0;
	while (!found && i<lw_nurbs_basket->count())
	{
		QListWidgetItem* item = lw_nurbs_basket->item(i);
		QString tinyname = item->text();

		found = searched_tinyname.compare(tinyname)==0;
		if (!found)
		{
			i++;
		}
	}
	if (found)
	{
		QListWidgetItem* item = lw_nurbs_basket->item(i);
		item->setSelected(true);
	}
	return found;
}

void GeotimeProjectManagerWidget::clear_nurbs_basket_gui_selection() {
	lw_nurbs_basket->clearSelection();
}


