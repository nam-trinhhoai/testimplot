#ifndef SURFACEMESH_H
#define SURFACEMESH_H

#include <Qt3DRender/QGeometryRenderer>
#include <QVector2D>
#include <QMatrix4x4>
#include <QVector>
//#include <QMutex>
#include "iimagepaletteholder.h"
#include "surfacemeshcacheutils.h"

namespace Qt3DCore {
class QBuffer;
class QAttribute;
class QGeometry;
}

class MinMax
{
public :
	int indice=0;
	int valeur=0;
	MinMax(int index, int value)
	{
		indice = index;
		valeur = value;
	}
};

class SymetricMatrix {

	public:

	// Constructor

	SymetricMatrix(double c=0) {
		//loopi(0,10)
		for(int i=0;i<10;i++)
			m[i] = c;
	}

	SymetricMatrix(	double m11, double m12, double m13, double m14,
			            double m22, double m23, double m24,
			                        double m33, double m34,
			                                    double m44) {
			 m[0] = m11;  m[1] = m12;  m[2] = m13;  m[3] = m14;
			              m[4] = m22;  m[5] = m23;  m[6] = m24;
			                           m[7] = m33;  m[8] = m34;
			                                        m[9] = m44;
	}

	// Make plane

	SymetricMatrix(double a,double b,double c,double d)
	{
		m[0] = a*a;  m[1] = a*b;  m[2] = a*c;  m[3] = a*d;
		             m[4] = b*b;  m[5] = b*c;  m[6] = b*d;
		                          m[7 ] =c*c; m[8 ] = c*d;
		                                       m[9 ] = d*d;
	}

	double operator[](int c) const { return m[c]; }

	// Determinant

	double det(	int a11, int a12, int a13,
				int a21, int a22, int a23,
				int a31, int a32, int a33)
	{
		double det =  m[a11]*m[a22]*m[a33] + m[a13]*m[a21]*m[a32] + m[a12]*m[a23]*m[a31]
					- m[a13]*m[a22]*m[a31] - m[a11]*m[a23]*m[a32]- m[a12]*m[a21]*m[a33];
		return det;
	}

	const SymetricMatrix operator+(const SymetricMatrix& n) const
	{
		return SymetricMatrix( m[0]+n[0],   m[1]+n[1],   m[2]+n[2],   m[3]+n[3],
						                    m[4]+n[4],   m[5]+n[5],   m[6]+n[6],
						                                 m[ 7]+n[ 7], m[ 8]+n[8 ],
						                                              m[ 9]+n[9 ]);
	}

	SymetricMatrix& operator+=(const SymetricMatrix& n)
	{
		 m[0]+=n[0];   m[1]+=n[1];   m[2]+=n[2];   m[3]+=n[3];
		 m[4]+=n[4];   m[5]+=n[5];   m[6]+=n[6];   m[7]+=n[7];
		 m[8]+=n[8];   m[9]+=n[9];
		return *this;
	}

	double m[10];
};

struct Ref { int tid,tvertex; };

class Vertex{
public:
	QVector3D p;

	int tstart,tcount;
	SymetricMatrix q;
	int border;
	QVector2D uvs;
	QVector3D n;
	Vertex()
	{

	}
};

class Triangle
{
public :

	int deleted;
	int dirty;
	double err[4];
	int v[3];
	int attr;
	QVector3D n;
	QVector2D uvs[3];
	Triangle()
	{
		deleted = 0;
	}
};

class SurfaceMesh : public Qt3DRender::QGeometryRenderer
{
    Q_OBJECT
    Q_PROPERTY(QVector2D dimensions READ dimensions WRITE setDimensions NOTIFY dimensionsChanged)
public:
    explicit SurfaceMesh(Qt3DCore::QNode *parent = nullptr);

    ~SurfaceMesh();

    void setDimensions(const QVector2D dimensions);
    QVector2D dimensions() const;

    void setTransform(const QMatrix4x4& transform);
    QMatrix4x4 transform() const;

    Qt3DCore::QBuffer* getVertexBuffer();
    Qt3DCore::QBuffer* getNormalBuffer();

    void computeNormals();

    void simplifyMesh(int target_count, double agressiveness=7);
    void update_mesh(int iteration);
    double vertex_error(SymetricMatrix q, double x, double y, double z);
    double calculate_error(int id_v1, int id_v2, QVector3D &p_result);
    bool flipped(QVector3D p,int i0,int i1,Vertex &v0,Vertex &v1,QVector<int> &deleted);
    void update_uvs(int i0,const Vertex &v,const QVector3D &p, QVector<int> &deleted);
    void update_triangles(int i0,Vertex  &v,QVector<int> &deleted,int &deleted_triangles);

    void compact_mesh();

    QVector3D barycentric(const QVector3D &p, const QVector3D &a, const QVector3D &b, const QVector3D &c);
    QVector2D interpolate(const QVector3D &p, const QVector3D &a, const QVector3D &b, const QVector3D &c, const QVector2D attrs[3]);


    template<typename IsoType>
    static void createCache(SurfaceMeshCache& cache,const IsoType* isobuffer ,int width, int depth ,float steps, float cubeOrigin,float cubeScale,QString path, QMatrix4x4 transform,int compression);

    template<typename IsoType>
    static void createObj(QString path,const IsoType* isobuffer, int width, int height, float steps, float cubeOrigin,float cubeScale,QMatrix4x4 objectTransform,int compression);
    static void loadObj(const char* filename,SurfaceMeshCache& cache);
    static void writeObj(const char* filename,QVector<Triangle> triangles,QVector<Vertex> vertices);
    void reaffectuv();
    void reloadFromCache(SurfaceMeshCache& cache,QMatrix4x4 sceneTransform);

    void initMesh(int width, int height,IImagePaletteHolder* palette, float heightThreshold,float cubeOrigin,float cubeScale, int simplifySteps);
    void init_obj(int width, int height,IImagePaletteHolder* palette, float heightThreshold,float cubeOrigin,float cubeScale,QString path, int simplifySteps, int compression,QMatrix4x4 sceneTransform ,QMatrix4x4 objectTranform);
    void load_obj(const char* filename, bool process_uv=false);
    void load_obj2(const char* filename, bool process_uv);//,float *vertex,float * texCoord, float * normals, unsigned int* indices);
    void write_obj(const char* filename);
    int getNumVertices();
    int getWidth();
    int getHeight();

    int getSensForIndex(int width,int height);
    void activateNullValue(float nullValue);
    void deactivateNullValue();
    float nullValue() const;
    bool isNullValueActive() const;


    QVector<QVector3D> normalsVec;
    QVector<QVector3D> m_listeNormals;

signals:
	void dimensionsChanged();
	void transformChanged();

private:
    void updateGeometry();
    void filterTrianglePoints(int i, const QVector3D& p1, const QVector3D& p2,
    		const QVector3D& p3, const uint* rawIndexArray, const float* rawVertexArray,
    		QVector<MinMax>& listeindex, QVector3D& n1);



    QVector2D m_dimensions;
    QMatrix4x4 m_transform;
    QMatrix4x4 m_sceneTransform;
    Qt3DCore::QBuffer *m_vertexBuffer;
    Qt3DCore::QAttribute *m_positionAttribute;

    Qt3DCore::QBuffer *m_texBuffer;
    Qt3DCore::QAttribute *m_textureAttribute;

    Qt3DCore::QBuffer *m_normalBuffer;
    Qt3DCore::QAttribute *m_normalAttribute;

    Qt3DCore::QBuffer *m_indexBuffer;
    Qt3DCore::QAttribute *m_indexAttribute;

    Qt3DCore::QGeometry *m_geometry;

    // Global Variables & Strctures
    	enum Attributes {
    		NONE,
    		NORMAL = 2,
    		TEXCOORD = 4,
    		COLOR = 8
    	};

    QVector<Triangle> triangles;
    QVector<Vertex> vertices;
	std::vector<Ref> refs;



	int m_steps;
	int m_processedSteps = -1;

    int m_width;
    int m_height;
    float m_nullValue;
    bool m_nullValueActive;
    float m_internalNullValue;

};

#include "surfacemesh.hpp"

#endif // VOLUMEBOUNDINGMESH_H
