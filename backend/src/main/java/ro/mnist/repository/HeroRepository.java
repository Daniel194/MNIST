package ro.mnist.repository;


import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.data.rest.core.annotation.RepositoryRestResource;
import ro.mnist.entity.Hero;

@RepositoryRestResource(collectionResourceRel = "heroes", path = "heroes")
public interface HeroRepository extends CrudRepository<Hero, Long> {

    @Query("FROM Hero h where LOWER(h.name) LIKE CONCAT('%', LOWER(:name), '%')")
    public Iterable<Hero> findByName(@Param("name") String name);

}