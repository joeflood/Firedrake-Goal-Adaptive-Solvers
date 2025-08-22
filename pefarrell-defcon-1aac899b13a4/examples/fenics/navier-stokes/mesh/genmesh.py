from dolfin import *

end = 150
ratio = 6

if __name__ == "__main__":
  from mshr import *
  inlet = Rectangle(Point(0, -1), Point(2.5, 1))
  main  = Rectangle(Point(2.5, -ratio), Point(end, +ratio))

  domain = inlet + main
  m = generate_mesh(domain, 500)

  File("mesh.xml.gz") << m
  plot(m, interactive=True)
