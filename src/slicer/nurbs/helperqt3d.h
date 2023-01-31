#pragma once

#include <QMaterial>
#include <QDiffuseSpecularMaterial>
#include <QGeometry>
#include <QGeometryRenderer>
#include <Qt3DCore/QTransform>
#include <QEntity>


#include <Qt3DRender/QEffect>
#include <Qt3DRender/QMaterial>
#include <Qt3DRender/QTechnique>
#include <Qt3DRender/QRenderPass>
#include <Qt3DRender/QShaderProgram>
#include <Qt3DRender/QGraphicsApiFilter>
#include <Qt3DRender/QParameter>
#include <QUrl>
#include <QVector2D>
#include <QVector3D>
#include <Qt3DRender/QCullFace>

#include <math.h>


//using namespace Qt3DRender;

class helperqt3d
{
public:


    class IsectPlane  {
        public:
            QVector3D getWorldPosition(double localX, double localY) const {return localX*xaxis+localY*yaxis+ pointinplane;}
            QVector2D getLocalPosition(QVector3D worldPos) const
            {
                QVector3D p = worldPos - pointinplane;
                double px = QVector3D::dotProduct(p,xaxis); // projects point onto fromXaxis and get its length, i.e. the x coordinate
                double py = QVector3D::dotProduct(p,yaxis); // projects point onto fromYaxis and get its length, i.e. the y coordinate
                return QVector2D(px, py);
            }
            QVector3D getNormal()
            {
            	return QVector3D::crossProduct(xaxis,yaxis);
            }

            bool isUndefined(){return xaxis==QVector3D(0,0,0);}
            QVector3D pointinplane, xaxis, yaxis;
    };

    static Qt3DExtras::QDiffuseSpecularMaterial* makeSimpleMaterial(QColor col)
    {
        Qt3DExtras::QDiffuseSpecularMaterial* mymat = new Qt3DExtras::QDiffuseSpecularMaterial();
        mymat->setAmbient(col);
        mymat->setDiffuse(col);
        mymat->setSpecular(QColor(0,0,0,0));
        return mymat;
    }
    static Qt3DExtras::QDiffuseSpecularMaterial* makeSimpleMaterial2(QColor col)
        {
            Qt3DExtras::QDiffuseSpecularMaterial* mymat = new Qt3DExtras::QDiffuseSpecularMaterial();
            mymat->setAmbient(QColor(col.red()*0.0f,col.green()*0.0f,col.blue()*0.0f,1.0f));
            mymat->setDiffuse(col);
            mymat->setSpecular(QColor(0,0,0,0));
            return mymat;
        }

    static Qt3DRender::QMaterial* makeShaderMaterial(QUrl vertexShaderfile, QUrl fragmentShaderfile)
    {
        // material sets effect, effect adds technique, technique adds pass, pass sets shaderprogram
        // further input uniforms to shader is added to material later so needs to be member
        Qt3DRender::QMaterial*      material      = new Qt3DRender::QMaterial();
        Qt3DRender::QEffect*        effect        = new Qt3DRender::QEffect();
        Qt3DRender::QTechnique*     glTechnique   = new Qt3DRender::QTechnique();
        Qt3DRender::QRenderPass*    glPass        = new Qt3DRender::QRenderPass();
        Qt3DRender::QShaderProgram* shaderprogram = new Qt3DRender::QShaderProgram();

        shaderprogram->setVertexShaderCode(Qt3DRender::QShaderProgram::loadSource(vertexShaderfile));
        shaderprogram->setFragmentShaderCode(Qt3DRender::QShaderProgram::loadSource(fragmentShaderfile));
        // todo: maybe more error output here and generalize it for all shaders
        if (shaderprogram->status() != Qt3DRender::QShaderProgram::Ready)    {
            //  if (shaderprogram->status() == QShaderProgram::NotReady) qDebug("QShaderProgram::NotReady");
            //  if (shaderprogram->status() == QShaderProgram::Error)    qDebug() << "QShaderProgram::Error" << shaderprogram->log();
        } else qDebug("QShaderProgram::Ready");

        glPass->setShaderProgram(shaderprogram);


        Qt3DRender::QCullFace* cullnone = new Qt3DRender::QCullFace();
        cullnone->setMode(Qt3DRender::QCullFace::NoCulling);
        glPass->addRenderState(cullnone);

        glTechnique->graphicsApiFilter()->setApi(Qt3DRender::QGraphicsApiFilter::OpenGL);
        glTechnique->graphicsApiFilter()->setMajorVersion(4);
        glTechnique->graphicsApiFilter()->setMinorVersion(3);
        glTechnique->graphicsApiFilter()->setProfile(Qt3DRender::QGraphicsApiFilter::CoreProfile);
        glTechnique->addRenderPass(glPass);

        effect->addTechnique(glTechnique);
        material->setEffect(effect);
        return material;
    }

    static void makeEntity(Qt3DCore::QEntity* entity, Qt3DRender::QMaterial* material,
                                               Qt3DCore::QGeometry* curveGeom, bool linestripOrPoints)
    {
        Qt3DRender::QGeometryRenderer* geomRenderer = new Qt3DRender::QGeometryRenderer();
        geomRenderer->setPrimitiveType(linestripOrPoints?Qt3DRender::QGeometryRenderer::LineStrip:Qt3DRender::QGeometryRenderer::Points);
        geomRenderer->setGeometry(curveGeom);

        makeEntity(entity,material, geomRenderer);
     }


     static void makeEntity(Qt3DCore::QEntity* entity,Qt3DRender::QMaterial* material, Qt3DRender::QGeometryRenderer* geomRenderer)
     {
         Qt3DCore::QTransform*      trans = new Qt3DCore::QTransform();
         entity->addComponent(geomRenderer);
         entity->addComponent(material);
         entity->addComponent(trans);
     }

     static void getPlanevectorsFromNormal(const QVector3D& normal, QVector3D& vector1InPlane, QVector3D& vector2InPlane)
     {
         QVector3D upvector(0,-1,0);
         float angle = acos(QVector3D::dotProduct(normal,upvector));

         if (angle<0.1) qDebug() << "tangent vector almost paralell with upvector, crossproduct might be inaccurate";

         vector1InPlane = QVector3D::crossProduct(-upvector,normal);   // plane should be tangential to curve as seen from above
         vector2InPlane = upvector;                                    // plane should be vertical

         vector1InPlane.normalize();
         vector2InPlane.normalize();
     }

};

